import { randomUUID } from "node:crypto";
import { verifyAgentSignature } from "@/lib/agents/signature";
import { isStellarAccount } from "@/lib/stellar-message";
import { getUserVKDFrom } from "@/lib/contract";
import {
  NETWORK_PASSPHRASE,
  STELLAR_NETWORK,
  getMarketContractId,
  isContractAddress,
  isMarketConfigured,
  validateNetworkPassphrase,
} from "@/lib/stellar";
import { publishReasoning } from "@/lib/reasoning/publish";
import {
  AGENT_ACTION_PAUSE, AGENT_API_ACTIONS, AGENT_API_VERSION, AGENT_FUNDED_ACTIONS, MAX_SIGNED_REQUEST_PAYLOAD_BYTES,
  agentRequestMessage, operatorProofMessage, validateAgentRequestEnvelope,
  type AgentApiAction, type SignedAgentRequest,
} from "@/lib/agents/api";
import { authenticateAgentRequest, requiresOwnerSignature } from "@/lib/agents/authenticate";
import {
  apiKeyPrefix, generateApiKey, hashApiKey, parseOverlapMs, type AgentApiKeyRecord,
} from "@/lib/agents/api-keys";
import {
  configuredSpender, currentPeriodStart, evaluateSpend, parseSpendPermissionGrant,
  type SpendPermissionGrant,
} from "@/lib/agents/spend-permissions";
import {
  getActiveSpendPermission, getSpentInPeriod, insertAgentApiKey, listAgentApiKeys,
  revokeAgentApiKey, revokeSpendPermission, setAgentApiKeyExpiry, upsertSpendPermission,
} from "@/lib/db";
import {
  AUTHORITY_LEVELS, REGISTRY_SCHEMA_VERSION, authorizeAction, defaultLimits,
  evaluateRegistrationReplay, revokeAgent, type AgentRecord, type AgentCapability,
} from "@/lib/agents/registry";
import { auditAgentRequest, loadAgent, loadIdempotentResponse, saveAgent, saveIdempotentResponse } from "@/lib/agents/store";
import { rejectReplayedSignedEnvelope } from "@/lib/agents/envelope-replay";
import { buildAgentDryRun } from "@/lib/agents/dry-run";
import { isFeatureEnabled, checkWriteAllowed, type Pausable } from "@/lib/ops/flags";
import { getUsdcBalanceUnits, usdcToUnits, parseUsdcAtomic } from "@/lib/usdc";
import { getAgentEarningsSummary } from "@/lib/db";
import { validateAgentApiRequest, schemaVersionHeaders } from "@/lib/api/schema";
import { gateOrPause, pausedCapabilityError, getCapabilityPauseDetail } from "@/lib/server/pause-registry";
import { apiError } from "@/lib/api/errors";
import { tracedRoute } from "@/lib/ops/trace-http";

export const dynamic = "force-dynamic";

/**
 * Wallet identities an agent record can carry.
 *
 * Stellar only now: `G` for an account, `C` for a contract account. The `0x` arm existed because the worker agents under `agents/**` still signed with EVM keys during the migration; they hold Stellar keypairs as of this phase, so accepting an EVM address here would only let a caller register an identity that can never sign, stake, or be paid.
 *
 * Nothing below normalises case. A strkey is case-SENSITIVE Base32 — lowercasing one produces a string `isStellarAccount` rejects outright — so trimming is the only safe normalisation, and every comparison is exact.
 */
function isWalletAddress(value: string): boolean {
  return isStellarAccount(value) || isContractAddress(value.trim());
}

/* Trim only: a Stellar strkey has no case-normalised form. */
function normalizeWallet(value: string): string {
  return value.trim();
}

function json(body: unknown, status = 200): Response {
  return Response.json(body, { status, headers: { "cache-control": "no-store", ...schemaVersionHeaders() } });
}

import { withRequestId } from "@/lib/api/errors";

/** Convert a structured ApiErrorResult from lib/api/errors into a Response. */
/** Map an authorizeAction() rejection to the typed agent API error envelope. */
function actionVerdictToError(
  gate: { reason?: string; detail?: string },
  capability: string,
): import("@/lib/api/errors").ApiErrorResult {
  switch (gate.reason) {
    case "platform_paused":
    case "paused":
      return apiError("agent_paused", gate.detail ?? "Agent or platform is paused");
    case "revoked":
      return apiError("agent_revoked", gate.detail ?? "Agent has been revoked");
    case "rate_limit_exceeded":
      return apiError("rate_limited", gate.detail ?? "Rate limit exceeded");
    case "missing_capability":
    case "insufficient_authority":
      return apiError("capability_missing", gate.detail ?? `Missing capability ${capability}`);
    default:
      return apiError("forbidden", gate.detail ?? gate.reason ?? "Action not authorized");
  }
}

function errorResponse(err: import("@/lib/api/errors").ApiErrorResult): Response {
  return Response.json(err.body, { status: err.status, headers: { ...err.headers, "cache-control": "no-store" } });
}

async function verify(address: string, message: string, signature: string): Promise<boolean> {
  return verifyAgentSignature({ address, message, signature });
}

async function audit(request: SignedAgentRequest, outcome: string, reason?: string) {
  await auditAgentRequest({
    requestId: randomUUID(), agentId: request.agentId, action: request.action,
    idempotencyKey: request.idempotencyKey, signedAt: request.signedAt,
    nonce: request.nonce, outcome, reason, createdAt: Date.now(),
  });
}

/**
 * Actions that put an owner's USDC at risk. Gated on byoa_funded_actions so the
 * launch-gate document and the code agree: until an operator enables it, an agent
 * can register, read and dry-run but cannot move money.
 *
 * The list itself lives in `lib/agents/api.ts` because the published wire contract
 * reads it from there — a second copy here and in the docs is a copy that drifts.
 */
const FUNDED_ACTIONS: readonly AgentApiAction[] = AGENT_FUNDED_ACTIONS;

async function register(request: SignedAgentRequest<Record<string, any>>): Promise<Response> {
  // Pause check first: "registration is paused" is more accurate than "not enabled"
  // when the feature is on but the capability is temporarily stopped.
  const pauseErr = gateOrPause({ feature: "byoa_registry", capability: "agent_registration" });
  if (pauseErr) return errorResponse(pauseErr);

  if (!isFeatureEnabled("byoa_registry")) {
    return json({ error: { message: "agent registration is not enabled" } }, 403);
  }
  const body = request.body;
  const owner = normalizeWallet(String(body.ownerWallet ?? ""));
  const operator = normalizeWallet(String(body.operatorWallet ?? ""));
  if (!isWalletAddress(owner) || !isWalletAddress(operator)) {
    return json({ error: { message: "invalid owner/operator wallet" } }, 400);
  }
  // The outer request is the owner's explicit grant over the exact record body.
  if (!(await verify(owner, agentRequestMessage(request), request.signature))) {
    return json({ error: { message: "owner signature rejected" } }, 401);
  }
  const operatorProof = operatorProofMessage(request.agentId, operator);
  if (!(await verify(operator, operatorProof, String(body.operatorSignature ?? "")))) {
    return json({ error: { message: "operator signature rejected" } }, 401);
  }

  // Safe client retries: same idempotency key returns the prior accepted body.
  if (request.idempotencyKey) {
    const prior = await loadIdempotentResponse(request.agentId, "register", request.idempotencyKey);
    if (prior) return json(prior.body, prior.status);
  }

  const payoutWallet = isWalletAddress(normalizeWallet(String(body.payoutWallet ?? "")))
    ? normalizeWallet(String(body.payoutWallet)) : owner;

  // Duplicate register with the same wallets is success, not conflict — check
  // identity BEFORE consuming the nonce so a retry of a successful admission is
  // not rejected as "nonce replay".
  const existing = await loadAgent(request.agentId);
  if (existing) {
    const replay = evaluateRegistrationReplay(existing, {
      ownerWallet: owner, operatorWallet: operator, payoutWallet,
    });
    if (replay.ok) {
      const payload = { agent: existing };
      if (request.idempotencyKey) {
        await saveIdempotentResponse(request.agentId, "register", request.idempotencyKey, payload, 200);
      }
      await audit(request, "registered_idempotent");
      return json(payload);
    }
    await audit(request, "rejected", replay.reason);
    return json({
      error: {
        message: "agent already registered",
        reason: replay.reason,
        detail: "a different owner, operator, or payout wallet claimed this agentId",
      },
    }, 409);
  }

  const registerReplay = await rejectReplayedSignedEnvelope({
    agentId: request.agentId,
    nonce: request.nonce,
    signedAt: request.signedAt,
  });
  if (registerReplay) {
    await audit(request, "rejected", "nonce_replay");
    return errorResponse(registerReplay);
  }
  const now = Date.now();
  const authority = Math.max(0, Math.min(4, Math.floor(Number(body.authorityLevel ?? 0)))) as AgentRecord["authorityLevel"];
  const requestedCapabilities = Array.isArray(body.capabilities) ? body.capabilities : [];
  const capabilities = requestedCapabilities.filter((c: unknown): c is AgentCapability =>
    typeof c === "string" && ["market_creator", "council_juror", "researcher", "copy_source", "x402_seller"].includes(c));
  const record: AgentRecord = {
    agentId: request.agentId,
    ownerWallet: owner,
    operatorWallet: operator,
    payoutWallet,
    authorityLevel: authority,
    capabilities,
    status: "active",
    createdAt: now,
    updatedAt: now,
  };
  await saveAgent(record);
  const payload = { agent: record };
  if (request.idempotencyKey) {
    await saveIdempotentResponse(request.agentId, "register", request.idempotencyKey, payload, 200);
  }
  await audit(request, "registered");
  return json(payload);
}

export const POST = trackedRoute(async (req: Request): Promise<Response> => {
  return json({ error: { message: "not implemented" } }, 501);
});
