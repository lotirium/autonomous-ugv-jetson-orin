import { CLOUD_API_BASE_URL } from "@/constants/env";
import type {
  ClaimRobotResponse,
  CloudRobot,
  CloudSession,
  CloudUser,
  NearbyRobotSummary,
} from "@/types/cloud";

const ensureBaseUrl = () => {
  if (!CLOUD_API_BASE_URL) {
    throw new Error(
      "Cloud API base URL is not configured. Set EXPO_PUBLIC_CLOUD_API_BASE_URL to continue."
    );
  }

  return CLOUD_API_BASE_URL;
};

const buildUrl = (path: string) => new URL(path, ensureBaseUrl()).toString();

const parseJson = async <T>(response: Response): Promise<T> => {
  const text = await response.text();
  if (!text) {
    return {} as T;
  }

  try {
    return JSON.parse(text) as T;
  } catch (error) {
    console.warn("Failed to parse JSON response", error, text);
    throw new Error("Unexpected response payload from cloud API.");
  }
};

const extractErrorMessage = async (response: Response) => {
  try {
    const payload = await parseJson<{ message?: string; error?: string }>(response);
    if (payload.message) {
      return payload.message;
    }
    if (payload.error) {
      return payload.error;
    }
  } catch (error) {
    const text = await response.text();
    if (text) {
      return text;
    }
  }

  return `Request failed with status ${response.status}.`;
};

const handleResponse = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return parseJson<T>(response);
};

const buildAuthHeaders = (token: string, includeJson = false) => {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token}`,
    Accept: "application/json",
  };

  if (includeJson) {
    headers["Content-Type"] = "application/json";
  }

  return headers;
};

export const fetchCurrentSession = async (
  token: string
): Promise<CloudSession> => {
  const response = await fetch(buildUrl("/session"), {
    method: "GET",
    headers: buildAuthHeaders(token),
  });

  const payload = await handleResponse<{
    user: CloudUser;
    token?: string;
  }>(response);

  return {
    token: payload.token ?? token,
    user: payload.user,
  };
};

export const fetchClaimedRobots = async (
  token: string
): Promise<CloudRobot[]> => {
  const response = await fetch(buildUrl("/robots/mine"), {
    method: "GET",
    headers: buildAuthHeaders(token),
  });

  const payload = await handleResponse<
    { robots?: CloudRobot[] } | CloudRobot[]
  >(response);

  if (Array.isArray(payload)) {
    return payload;
  }

  return payload.robots ?? [];
};

export const fetchUnclaimedRobots = async (
  token: string
): Promise<NearbyRobotSummary[]> => {
  const response = await fetch(buildUrl("/robots/unclaimed"), {
    method: "GET",
    headers: buildAuthHeaders(token),
  });

  const payload = await handleResponse<
    { robots?: NearbyRobotSummary[] } | NearbyRobotSummary[]
  >(response);

  if (Array.isArray(payload)) {
    return payload;
  }

  return payload.robots ?? [];
};

export const claimRobot = async (
  token: string,
  payload: { serial: string; pin: string }
): Promise<ClaimRobotResponse> => {
  const response = await fetch(buildUrl("/robots/claim"), {
    method: "POST",
    headers: buildAuthHeaders(token, true),
    body: JSON.stringify(payload),
  });

  const result = await handleResponse<ClaimRobotResponse>(response);
  if (!result.robot) {
    throw new Error("Claim response did not include robot details.");
  }

  return result;
};

export const releaseRobot = async (
  token: string,
  robotId: string
): Promise<void> => {
  const response = await fetch(buildUrl(`/robots/${robotId}/release`), {
    method: "POST",
    headers: buildAuthHeaders(token, true),
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }
};
