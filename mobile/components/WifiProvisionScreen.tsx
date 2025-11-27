import * as Network from "expo-network";
import { useRouter } from "expo-router";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Modal,
  PermissionsAndroid,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { ThemedText } from "@/components/themed-text";
import { ThemedView } from "@/components/themed-view";
import { DEFAULT_ROBOT_BASE_URL, useRobot } from "@/context/robot-provider";
import { createRobotApi } from "@/services/robot-api";
import { checkAllRobotsStatus } from "@/services/robot-status-check";
import { RobotStatusCheck as RobotStatusCheckType } from "@/services/robot-storage";
import { Image } from "expo-image";
import { IconSymbol } from "./ui/icon-symbol";
import WifiManager from "react-native-wifi-reborn";

const deriveHost = (value: string | null | undefined) => {
  if (!value) {
    return null;
  }

  try {
    const parsed = new URL(value.includes("://") ? value : `http://${value}`);
    return parsed.host || parsed.hostname || null;
  } catch (error) {
    console.warn("Unable to parse host from value", value, error);
    return value;
  }
};

const formatBaseUrl = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed.replace(/\/$/, "");
  }

  return `http://${trimmed.replace(/\/$/, "")}`;
};

const StatusPill = ({ color, label }: { color: string; label: string }) => (
  <View style={styles.statusPill}>
    <View style={[styles.statusDot, { backgroundColor: color }]} />
    <ThemedText style={styles.statusPillText}>{label}</ThemedText>
  </View>
);

const getSignalStrengthInfo = (rssi?: number | null) => {
  if (typeof rssi !== "number") {
    return { label: "Unknown", color: "#67686C" };
  }
  if (rssi >= -60) {
    return { label: "Strong", color: "#1DD1A1" };
  }
  if (rssi >= -75) {
    return { label: "Medium", color: "#FBBF24" };
  }
  return { label: "Weak", color: "#F87171" };
};

/**
 * Wi-Fi Provision Screen Component
 * Guides the user through connecting to the robot hotspot and
 * managing the Wi-Fi connection.
 */
export function WifiProvisionScreen() {
  const {
    refreshStatus,
    status,
    statusError,
    baseUrl,
    setBaseUrl,
    connectToStoredRobot,
    api,
  } = useRobot();

  const router = useRouter();

  const [selectedNetwork, setSelectedNetwork] = useState<string>("");
  const [ssid, setSsid] = useState("");
  const [password, setPassword] = useState("");
  const [isCheckingNetwork, setIsCheckingNetwork] = useState(false);
  const [isManualModalVisible, setIsManualModalVisible] = useState(false);
  const [manualIpInput, setManualIpInput] = useState("");
  const [manualIpEdited, setManualIpEdited] = useState(false);
  const [manualConnectError, setManualConnectError] = useState<string | null>(
    null
  );
  const [isManualConnecting, setIsManualConnecting] = useState(false);
  const [savedRobots, setSavedRobots] = useState<RobotStatusCheckType[]>([]);
  const [isCheckingRobots, setIsCheckingRobots] = useState(false);
  const [phoneWifiNetworks, setPhoneWifiNetworks] = useState<
    { ssid: string; rssi: number }[]
  >([]);
  const [isScanningPhoneWifi, setIsScanningPhoneWifi] = useState(false);
  const [isConfiguringWifi, setIsConfiguringWifi] = useState(false);

  const isCheckingNetworkRef = useRef(false);
  const refreshStatusRef = useRef(refreshStatus);

  // Keep ref updated
  useEffect(() => {
    refreshStatusRef.current = refreshStatus;
  }, [refreshStatus]);

  useEffect(() => {
    if (manualIpEdited) {
      return;
    }

    const fallback =
      status?.network?.ip ||
      deriveHost(baseUrl) ||
      deriveHost(DEFAULT_ROBOT_BASE_URL);

    if (fallback) {
      setManualIpInput(fallback);
    }
  }, [baseUrl, manualIpEdited, status?.network?.ip]);

  /**
   * Scan for Wi-Fi networks visible to this device (used to find the robot hotspot)
   */
  const handleScanNetworks = useCallback(async () => {
    if (Platform.OS !== "android") {
      Alert.alert(
        "Scanning not supported",
        "Wi-Fi scanning is only available on Android. Please manually connect your phone to the robot hotspot, then continue."
      );
      return;
    }

    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      {
        title: "Location permission required",
        message:
          "We need location access to scan Wi-Fi networks near this device.",
        buttonPositive: "Allow",
        buttonNegative: "Deny",
      }
    );

    if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
      Alert.alert(
        "Permission required",
        "Enable location access to scan nearby Wi-Fi networks."
      );
      return;
    }

    setIsScanningPhoneWifi(true);
    setPhoneWifiNetworks([]);

    try {
      const scanResults = await WifiManager.reScanAndLoadWifiList();

      const networks = (scanResults || []).map((network: any) => ({
        ssid: network?.SSID || network?.ssid || "",
        rssi:
          typeof network?.level === "number"
            ? network.level
            : typeof network?.signalStrength === "number"
              ? network.signalStrength
              : -100,
      }));

      const filtered = networks.filter((network) => network.ssid);
      setPhoneWifiNetworks(filtered);

      if (filtered.length === 0) {
        Alert.alert(
          "No Networks Found",
          "No Wi-Fi networks were detected near this device. Move closer to your router or robot hotspot and try again."
        );
      }
    } catch (error) {
      Alert.alert(
        "Scan Error",
        error instanceof Error
          ? error.message
          : "Failed to scan nearby Wi-Fi networks from this device"
      );
    } finally {
      setIsScanningPhoneWifi(false);
    }
  }, []);

  /**
   * Get phone's current network info
   */
  const refreshPhoneNetwork = useCallback(async () => {
    try {
      const networkState = await Network.getNetworkStateAsync();
      const ipAddress = await Network.getIpAddressAsync();
      const resolvedIp =
        ipAddress && ipAddress !== "0.0.0.0" ? ipAddress : null;

      // Try to get SSID (may not be available on all platforms)
      let ssidValue: string | null = null;
      if (networkState.type === "WIFI") {
        ssidValue = null;
      }

      console.log("Phone network:", { ssidValue, ipAddress: resolvedIp });
    } catch (error) {
      console.warn("Failed to get phone network info:", error);
    }
  }, []);

  /**
   * Check if robot is on the same network as phone
   */
  const checkSameNetwork = useCallback(async () => {
    if (isCheckingNetworkRef.current) {
      return;
    }

    isCheckingNetworkRef.current = true;
    setIsCheckingNetwork(true);
    try {
      await refreshStatusRef.current();

      setTimeout(() => {
        setIsCheckingNetwork(false);
        isCheckingNetworkRef.current = false;
      }, 500);
    } catch (error) {
      console.log("Robot not on same network:", error);
      setIsCheckingNetwork(false);
      isCheckingNetworkRef.current = false;
    }
  }, []);

  /**
   * Step 3: Send Wi-Fi configuration
   */
  const handleSendConfig = useCallback(async () => {
    if (!ssid.trim()) {
      Alert.alert("Invalid Input", "Please enter a Wi-Fi network name (SSID)");
      return;
    }

    try {
      if (!api) {
        throw new Error("Robot API unavailable. Connect to the hotspot first.");
      }

      setIsConfiguringWifi(true);
      await api.connectWifi({ ssid: ssid.trim(), password });
      await refreshStatus();
      await checkSameNetwork();
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Failed to send Wi-Fi configuration";
      Alert.alert("Configuration Error", message);
    } finally {
      setIsConfiguringWifi(false);
    }
  }, [api, ssid, password, refreshStatus, checkSameNetwork]);

  const handleManualConnect = useCallback(async () => {
    const formatted = formatBaseUrl(manualIpInput);
    if (!formatted) {
      setManualConnectError("Enter a valid IP address or URL.");
      return;
    }

    setManualConnectError(null);
    setIsManualConnecting(true);

    try {
      const probeApi = createRobotApi(formatted, 4000, null);
      await probeApi.fetchHealth();

      // First, check if this is a previously connected robot
      const connected = await connectToStoredRobot(formatted);
      if (connected) {
        // Successfully connected to stored robot - skip additional setup
        await refreshStatus();
        setIsManualModalVisible(false);
        router.push("/(tabs)/home");
        return;
      }

      // Not a stored robot, proceed with normal connection
      setBaseUrl(formatted);
      await refreshStatus();
      setIsManualModalVisible(false);
      router.replace("/wifi");
    } catch (error) {
      setManualConnectError(
        error instanceof Error
          ? error.message
          : "Failed to reach the robot. Double-check the IP and try again."
      );
    } finally {
      setIsManualConnecting(false);
    }
  }, [manualIpInput, refreshStatus, router, setBaseUrl, connectToStoredRobot]);

  // Load and check saved robots on mount
  useEffect(() => {
    let mounted = true;
    void (async () => {
      try {
        setIsCheckingRobots(true);
        const checks = await checkAllRobotsStatus();
        if (mounted) {
          setSavedRobots(checks);
        }
      } catch (error) {
        console.warn("Failed to check saved robots", error);
      } finally {
        if (mounted) {
          setIsCheckingRobots(false);
        }
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  // Refresh phone network on mount (only once)
  useEffect(() => {
    let mounted = true;
    void (async () => {
      try {
        await refreshPhoneNetwork();
      } catch (error) {
        if (mounted) {
          console.warn("Failed to refresh phone network on mount:", error);
        }
      }
    })();
    return () => {
      mounted = false;
    };
  }, [refreshPhoneNetwork]);

  const wifiConnectionStatus = useMemo(() => {
    if (status?.network?.ip) {
      return { label: "Connected", color: "#1DD1A1" };
    }
    if (statusError) {
      return { label: "Offline", color: "#F87171" };
    }
    return { label: "Pending", color: "#FBBF24" };
  }, [status?.network?.ip, statusError]);

  const overlayVisible =
    isManualConnecting ||
    isCheckingNetwork ||
    isConfiguringWifi;

  const overlayMessage = (() => {
    if (isManualConnecting) {
      return "Establishing connection to your robot";
    }
    if (isConfiguringWifi) {
      return "Sending Wi-Fi credentials";
    }
    if (isCheckingNetwork) {
      return "Checking network connection";
    }
    return "Working";
  })();

  // Reconnect to a saved robot
  const handleReconnectToRobot = useCallback(
    async (robotCheck: RobotStatusCheckType) => {
      const robot = robotCheck.robot;
      const baseUrl =
        robot.baseUrl ||
        (robot.last_ip ? `http://${robot.last_ip}:8000` : null);

      if (!baseUrl) {
        Alert.alert("Error", "Robot IP address not available.");
        return;
      }

      setIsManualConnecting(true);
      try {
        if (robotCheck.status === "ready") {
          // Previously saved robot, connect directly
          const connected = await connectToStoredRobot(baseUrl);
          if (connected) {
            await refreshStatus();
            router.push("/(tabs)/home");
          }
        } else {
          // Needs setup or offline, go to connection flow
          setBaseUrl(baseUrl);
          await refreshStatus();
          router.push("/connection");
        }
      } catch (error) {
        Alert.alert(
          "Connection Failed",
          error instanceof Error ? error.message : "Failed to connect to robot"
        );
      } finally {
        setIsManualConnecting(false);
      }
    },
    [connectToStoredRobot, refreshStatus, router, setBaseUrl]
  );

  // Get status badge for saved robot
  const getRobotStatusBadge = (robotCheck: RobotStatusCheckType) => {
    switch (robotCheck.status) {
      case "ready":
        return { label: "Ready", color: "#1DD1A1" };
      case "needs_repair":
        return { label: "Needs setup", color: "#FBBF24" };
      case "offline":
        return { label: "Offline", color: "#F87171" };
      default:
        return { label: "Unknown", color: "#67686C" };
    }
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={["top"]}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <ThemedView style={styles.container}>
          <View style={styles.header}>
            <ThemedText type="title" style={styles.title}>
              Connect to a robot
            </ThemedText>
          </View>

          {statusError ? (
            <ThemedView style={styles.errorCard}>
              <ThemedText style={styles.errorText}>{statusError}</ThemedText>
            </ThemedView>
          ) : null}


          <View>
            <View style={styles.sectionHeader}>
              <View>
                <ThemedText style={styles.sectionTitle}>
                  Wi-Fi hotspot
                </ThemedText>
                <ThemedText style={styles.sectionHint}>
                  Connect your phone to the robot hotspot, then pick the Wi-Fi
                  network the robot should join.
                </ThemedText>
              </View>
              <StatusPill
                color={wifiConnectionStatus.color}
                label={wifiConnectionStatus.label}
              />
            </View>

            <View style={styles.sectionCard}>
              <View style={styles.blockHeader}>
                <ThemedText style={styles.blockTitle}>Available networks</ThemedText>
                <Pressable
                  style={[
                    styles.scanButton,
                    (isScanningPhoneWifi || isConfiguringWifi) && styles.disabledPrimary,
                  ]}
                  onPress={handleScanNetworks}
                  disabled={isScanningPhoneWifi || isConfiguringWifi}
                >
                  {isScanningPhoneWifi ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <IconSymbol
                      size={20}
                      name="arrow.trianglehead.2.clockwise"
                      color="#fff"
                    />
                  )}
                </Pressable>
              </View>

              {phoneWifiNetworks.length === 0 && !isScanningPhoneWifi ? (
                <ThemedText style={styles.emptyStateText}>
                  Scan to find Wi-Fi networks visible to this device.
                </ThemedText>
              ) : null}

              {phoneWifiNetworks.length > 0 && (
                <View style={styles.wifiList}>
                  {phoneWifiNetworks.map((network, index) => {
                    const signalInfo = getSignalStrengthInfo(network.rssi);
                    const isSelected = selectedNetwork === network.ssid;
                    return (
                      <Pressable
                        key={`${network.ssid}-${index}`}
                        style={[styles.wifiItem, isSelected && styles.deviceSelected]}
                        onPress={() => {
                          setSelectedNetwork(network.ssid);
                          setSsid(network.ssid);
                        }}
                      >
                        <ThemedText style={styles.wifiSsid}>
                          {network.ssid || "Unknown"}
                        </ThemedText>
                        <View
                          style={[
                            styles.signalBadge,
                            { borderColor: signalInfo.color },
                          ]}
                        >
                          <View
                            style={[
                              styles.signalDot,
                              { backgroundColor: signalInfo.color },
                            ]}
                          />
                          <ThemedText
                            style={[
                              styles.signalBadgeText,
                              { color: signalInfo.color },
                            ]}
                          >
                            {signalInfo.label}
                          </ThemedText>
                        </View>
                      </Pressable>
                    );
                  })}
                </View>
              )}

              <View style={styles.form}>
                <View>
                  <ThemedText style={styles.label}>Wi-Fi network</ThemedText>
                  <TextInput
                    style={styles.input}
                    value={ssid}
                    onChangeText={setSsid}
                    placeholder="Network SSID"
                    placeholderTextColor="#6B7280"
                    autoCapitalize="none"
                  />
                </View>
                <View>
                  <ThemedText style={styles.label}>Password (optional)</ThemedText>
                  <TextInput
                    style={styles.input}
                    value={password}
                    onChangeText={setPassword}
                    placeholder="Password"
                    placeholderTextColor="#6B7280"
                    secureTextEntry
                  />
                </View>
                <Pressable
                  style={[
                    styles.primaryButton,
                    isConfiguringWifi && styles.disabledPrimary,
                  ]}
                  onPress={handleSendConfig}
                  disabled={isConfiguringWifi}
                >
                  {isConfiguringWifi ? (
                    <ActivityIndicator color="#04110B" />
                  ) : (
                    <ThemedText style={styles.primaryButtonText}>
                      Send Wi-Fi credentials
                    </ThemedText>
                  )}
                </Pressable>
              </View>
            </View>

            {/* Previously Connected Robots Card */}
            {savedRobots.length !== 0 && <ThemedView style={styles.sectionCard}>
              <ThemedText style={styles.sectionTitle}>
                Previously connected robots
              </ThemedText>
              {isCheckingRobots ? (
                <View style={styles.inlineStatus}>
                  <ActivityIndicator size="small" color="#1DD1A1" />
                  <ThemedText style={styles.statusLabelText}>
                    Checking robots...
                  </ThemedText>
                </View>
              ) : savedRobots.length === 0 ? (
                <ThemedText style={styles.emptyStateText}>
                  No previously connected robots
                </ThemedText>
              ) : (
                <View style={styles.robotList}>
                  {savedRobots.map((robotCheck) => {
                    const robot = robotCheck.robot;
                    const statusBadge = getRobotStatusBadge(robotCheck);
                    const displayName =
                      robot.name ||
                      `(${robot.last_wifi_ssid || robot.last_ip || "unknown"
                      })`;
                    const subtitle =
                      robotCheck.status === "ready"
                        ? robotCheck.robotStatus?.wifi?.ssid
                          ? `${robotCheck.robotStatus.wifi.ssid} – Tap to connect`
                          : "Tap to connect"
                        : robotCheck.status === "needs_repair"
                          ? "Needs Wi-Fi setup"
                          : robot.last_wifi_ssid
                            ? `Last seen: ${robot.last_wifi_ssid} (${robot.last_ip || "offline"
                            })`
                            : robot.last_ip
                              ? `Last IP: ${robot.last_ip}`
                              : "Offline";

                    return (
                      <Pressable
                        key={robot.robot_id}
                        style={styles.robotItem}
                        onPress={() => handleReconnectToRobot(robotCheck)}
                      >
                        <View style={styles.robotItemContent}>
                          <View style={styles.robotItemHeader}>
                            <ThemedText style={styles.robotName}>
                              {displayName}
                            </ThemedText>
                            <StatusPill
                              color={statusBadge.color}
                              label={statusBadge.label}
                            />
                          </View>
                          <ThemedText style={styles.robotSubtitle}>
                            {subtitle}
                          </ThemedText>
                        </View>
                        <IconSymbol
                          size={20}
                          name="chevron.right"
                          color="#67686C"
                        />
                      </Pressable>
                    );
                  })}
                </View>
              )}
            </ThemedView>}
          </View>
        </ThemedView>
      </ScrollView>
      <View style={styles.bottomActionContainer}>
        <Pressable
          style={styles.connectIpButton}
          onPress={() => {
            setManualConnectError(null);
            setIsManualModalVisible(true);
          }}
        >
          <ThemedText style={styles.connectIpText}>
            Connect to a specific IP
          </ThemedText>
        </Pressable>
        <Image
          source={require("@/assets/images/head.png")}
          style={styles.robotImage}
          contentFit="contain"
        />
      </View>

      <Modal
        visible={isManualModalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setIsManualModalVisible(false)}
      >
        <View style={styles.modalBackdrop}>
          <KeyboardAvoidingView
            behavior={Platform.OS === "ios" ? "padding" : undefined}
            style={styles.modalWrapper}
          >
            <ThemedView style={styles.modalCard}>
              <ThemedText type="subtitle" style={styles.modalTitle}>
                Connect to a specific IP
              </ThemedText>
              <ThemedText style={styles.modalHint}>
                Enter the robot’s IP address or base URL.
              </ThemedText>
              <TextInput
                style={styles.input}
                value={manualIpInput}
                onChangeText={(text) => {
                  setManualIpInput(text);
                  setManualIpEdited(true);
                  setManualConnectError(null);
                }}
                placeholder="10.0.0.5 or https://robot.local"
                placeholderTextColor="#6B7280"
                autoCapitalize="none"
                autoCorrect={false}
              />
              {manualConnectError ? (
                <ThemedText style={styles.errorText}>
                  {manualConnectError}
                </ThemedText>
              ) : null}
              <View style={styles.modalActions}>
                <Pressable
                  style={[styles.secondaryButton, styles.outlineButton]}
                  onPress={() => setIsManualModalVisible(false)}
                  disabled={isManualConnecting}
                >
                  <ThemedText style={styles.outlineButtonText}>
                    Cancel
                  </ThemedText>
                </Pressable>
                <Pressable
                  style={[
                    styles.primaryButton,
                    isManualConnecting && styles.disabledPrimary,
                  ]}
                  onPress={handleManualConnect}
                  disabled={isManualConnecting}
                >
                  {isManualConnecting ? (
                    <ActivityIndicator color="#04110B" />
                  ) : (
                    <ThemedText style={styles.primaryButtonText}>
                      Connect
                    </ThemedText>
                  )}
                </Pressable>
              </View>
            </ThemedView>
          </KeyboardAvoidingView>
        </View>
      </Modal>

      <Modal visible={overlayVisible} transparent animationType="fade">
        <View style={styles.overlayBackdrop}>
          <View style={styles.overlayCard}>
            <ActivityIndicator size="large" color="#1DD1A1" />
            <ThemedText style={styles.overlayText}>{overlayMessage}</ThemedText>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  robotImage: {
    width: "100%",
    aspectRatio: 375 / 100,
  },
  safeArea: {
    flex: 1,
    backgroundColor: "#161616",
    padding: 24,
    paddingBottom: 0,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 48,
  },
  container: {
    flex: 1,
    gap: 24,
    marginVertical: "auto",
    backgroundColor: "#161616",
  },
  header: {
    gap: 12,
  },
  title: {
    fontSize: 32,
    lineHeight: 36,
  },
  subtitle: {
    color: "#67686C",
  },
  errorCard: {
    padding: 16,
    borderWidth: 1,
    borderColor: "#F87171",
    backgroundColor: "#1F1A1A",
  },
  errorText: {
    color: "#F87171",
  },
  sectionCard: {
    padding: 20,
    borderWidth: 1,
    borderColor: "#202020",
    backgroundColor: "#1C1C1C",
    marginVertical: 8,
    gap: 20,
  },
  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
  },
  sectionTitle: {
    fontSize: 18,
    fontFamily: "JetBrainsMono_600SemiBold",
    color: "#67686C",
  },
  sectionHint: {
    color: "#67686C",
    marginTop: 4,
    fontSize: 14,
  },
  blockHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  blockTitle: {
    fontSize: 14,
    color: "#D1D5DB",
    fontFamily: "JetBrainsMono_600SemiBold",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  inlineStatus: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  deviceList: {
    gap: 12,
  },
  deviceItem: {
    gap: 12,
  },
  deviceHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  deviceSelected: {
    borderColor: "#1DD1A1",
  },
  deviceName: {
    color: "#F9FAFB",
    fontSize: 16,
    fontFamily: "JetBrainsMono_600SemiBold",
  },
  deviceId: {
    color: "#6B7280",
    fontSize: 12,
  },
  deviceMeta: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  signalBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  signalBadgeText: {
    fontSize: 12,
    fontFamily: "JetBrainsMono_600SemiBold",
  },
  signalDot: {
    width: 8,
    height: 8,
  },
  signalRssiText: {
    color: "#9CA3AF",
    fontSize: 12,
    fontFamily: "JetBrainsMono_500Medium",
  },
  connectedBadge: {
    color: "#1DD1A1",
    fontFamily: "JetBrainsMono_600SemiBold",
    fontSize: 14,
  },
  emptyStateText: {
    color: "#67686C",
    fontStyle: "italic",
  },
  primaryButton: {
    backgroundColor: "#1DD1A1",
    paddingVertical: 12,
    paddingInline: 16,
    alignItems: "center",
  },
  scanButton: {
    alignItems: "center",
    justifyContent: "center",
  },
  disabledPrimary: {
    opacity: 0.5,
  },
  primaryButtonText: {
    color: "#04110B",
    fontFamily: "JetBrainsMono_600SemiBold",
    fontSize: 16,
  },
  secondaryButton: {
    paddingVertical: 12,
    paddingInline: 16,
    alignItems: "center",
    justifyContent: "center",
  },
  secondaryButtonText: {
    color: "#E5E7EB",
    fontSize: 16,
  },
  outlineButton: {
    borderWidth: 1,
    borderColor: "#202020",
    backgroundColor: "transparent",
  },
  outlineButtonText: {
    color: "#E5E7EB",
    fontFamily: "JetBrainsMono_600SemiBold",
  },
  buttonDisabled: {
    opacity: 0.4,
  },
  statusBoard: {
    borderWidth: 1,
    borderColor: "#202020",
    backgroundColor: "#1B1B1B",
    padding: 16,
    gap: 12,
  },
  statusRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  statusLabel: {
    color: "#67686C",
    fontSize: 14,
  },
  statusValue: {
    color: "#F9FAFB",
    fontFamily: "JetBrainsMono_600SemiBold",
  },
  statusValueRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  statusIndicator: {
    width: 10,
    height: 10,
  },
  infoCard: {
    padding: 16,

    backgroundColor: "#202020",
  },
  infoText: {
    color: "#E5E7EB",
    fontSize: 14,
  },
  form: {
    gap: 16,
  },
  label: {
    fontSize: 14,
    color: "#D1D5DB",
    fontFamily: "JetBrainsMono_600SemiBold",
  },
  input: {
    backgroundColor: "#1C1C1C",
    borderWidth: 1,
    borderColor: "#202020",

    padding: 16,
    color: "#F9FAFB",
    fontSize: 16,
    fontFamily: "monospace",
  },
  statusLabelText: {
    color: "#D1D5DB",
    fontSize: 12,
  },
  previousItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  previousDot: {
    width: 6,
    height: 6,
    backgroundColor: "#4B5563",
  },
  previousText: {
    color: "#D1D5DB",
  },
  provisionStatusRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  bottomActionContainer: {
    position: "fixed",
    bottom: 0,
    paddingTop: 12,
    backgroundColor: "#161616",
    gap: 36,
  },
  connectIpButton: {
    borderWidth: 1,
    borderColor: "#202020",
    paddingVertical: 16,
    alignItems: "center",
    backgroundColor: "#1B1B1B",
  },
  connectIpText: {
    color: "#E5E7EB",
    fontFamily: "JetBrainsMono_600SemiBold",
  },
  statusPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
    backgroundColor: "#202020",
  },
  statusDot: {
    width: 8,
    height: 8,
  },
  statusPillText: {
    color: "#F9FAFB",
    fontSize: 12,
    fontFamily: "JetBrainsMono_600SemiBold",
  },
  subsectionTitle: {
    fontSize: 14,
    fontFamily: "JetBrainsMono_600SemiBold",
    color: "#D1D5DB",
    marginBottom: 8,
  },
  wifiList: {
    gap: 12,
    marginTop: 12,
  },
  wifiItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: "#202020",
    backgroundColor: "#1B1B1B",
  },
  wifiSsid: {
    color: "#F9FAFB",
    fontSize: 16,
    fontFamily: "JetBrainsMono_600SemiBold",
  },
  robotList: {
    gap: 12,
    marginTop: 12,
  },
  robotItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: "#202020",
    backgroundColor: "#1B1B1B",
  },
  robotItemContent: {
    flex: 1,
    gap: 4,
  },
  robotItemHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 12,
  },
  robotName: {
    color: "#F9FAFB",
    fontSize: 16,
    fontFamily: "JetBrainsMono_600SemiBold",
    flex: 1,
  },
  robotSubtitle: {
    color: "#67686C",
    fontSize: 12,
    fontFamily: "JetBrainsMono_400Regular",
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.7)",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  modalWrapper: {
    width: "100%",
  },
  modalCard: {
    padding: 20,
    borderWidth: 1,
    borderColor: "#202020",
    backgroundColor: "#161616",
    gap: 16,
  },
  modalTitle: {
    fontSize: 20,
    color: "#F9FAFB",
  },
  modalHint: {
    color: "#67686C",
  },
  modalActions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 12,
  },
  overlayBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.8)",
    justifyContent: "center",
    alignItems: "center",
  },
  overlayCard: {
    padding: 24,
    backgroundColor: "#161616",
    borderWidth: 1,
    borderColor: "#202020",
    alignItems: "center",
    gap: 16,
  },
  overlayText: {
    color: "#F9FAFB",
    fontSize: 16,
    textAlign: "center",
  },
});
