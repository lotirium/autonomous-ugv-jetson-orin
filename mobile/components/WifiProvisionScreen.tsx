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
  Animated,
  Easing,
  KeyboardAvoidingView,
  Modal,
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
import { useRovyBle } from "@/hooks/use-rovy-ble";
import { createRobotApi } from "@/services/robot-api";
import { checkAllRobotsStatus } from "@/services/robot-status-check";
import { RobotStatusCheck as RobotStatusCheckType } from "@/services/robot-storage";
import type { RovyDevice } from "@/services/rovy-ble";
import { Image } from "expo-image";
import { IconSymbol } from "./ui/icon-symbol";

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

const formatRssiValue = (rssi?: number | null) => {
  if (typeof rssi !== "number") {
    return "Signal unknown";
  }
  return `${rssi} dBm`;
};

/**
 * Wi-Fi Provision Screen Component
 * Guides the user through finding the robot over BLE and
 * managing the Wi-Fi connection.
 */
export function WifiProvisionScreen() {
  const {
    isScanning,
    isConnecting,
    isSendingConfig,
    isConnected,
    wifiStatus,
    error,
    scanForRovy,
    connectToRovy,
    sendWifiConfig,
    disconnect,
  } = useRovyBle();

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

  const [devices, setDevices] = useState<RovyDevice[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<RovyDevice | null>(null);
  const [ssid, setSsid] = useState("");
  const [password, setPassword] = useState("");
  const [showWifiConfig, setShowWifiConfig] = useState(false);
  const [isCheckingNetwork, setIsCheckingNetwork] = useState(false);
  const [isOnSameNetwork, setIsOnSameNetwork] = useState<boolean | null>(null);
  const [isRobotConnecting, setIsRobotConnecting] = useState(false);
  const [isManualModalVisible, setIsManualModalVisible] = useState(false);
  const [manualIpInput, setManualIpInput] = useState("");
  const [manualIpEdited, setManualIpEdited] = useState(false);
  const [manualConnectError, setManualConnectError] = useState<string | null>(
    null
  );
  const [isManualConnecting, setIsManualConnecting] = useState(false);
  const [savedRobots, setSavedRobots] = useState<RobotStatusCheckType[]>([]);
  const [isCheckingRobots, setIsCheckingRobots] = useState(false);
  const [robotWifiNetworks, setRobotWifiNetworks] = useState<
    { ssid: string; rssi: number }[]
  >([]);
  const [isScanningRobotWifi, setIsScanningRobotWifi] = useState(false);
  const scanRotationValue = useRef(new Animated.Value(0)).current;
  const scanRotationLoop = useRef<Animated.CompositeAnimation | null>(null);
  const scanRotation = useMemo(
    () =>
      scanRotationValue.interpolate({
        inputRange: [0, 1],
        outputRange: ["0deg", "360deg"],
      }),
    [scanRotationValue]
  );

  const isCheckingNetworkRef = useRef(false);
  const refreshStatusRef = useRef(refreshStatus);

  // Keep ref updated
  useEffect(() => {
    refreshStatusRef.current = refreshStatus;
  }, [refreshStatus]);

  useEffect(() => {
    if (isScanning) {
      scanRotationLoop.current = Animated.loop(
        Animated.timing(scanRotationValue, {
          toValue: 1,
          duration: 1200,
          easing: Easing.linear,
          useNativeDriver: true,
        })
      );
      scanRotationLoop.current.start();
    } else {
      scanRotationLoop.current?.stop();
      scanRotationValue.stopAnimation(() => {
        scanRotationValue.setValue(0);
      });
      scanRotationLoop.current = null;
    }

    return () => {
      scanRotationLoop.current?.stop();
      scanRotationLoop.current = null;
    };
  }, [isScanning, scanRotationValue]);

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
   * Step 1: Scan for ROVY devices
   */
  const handleScan = useCallback(async () => {
    try {
      setDevices([]);
      setSelectedDevice(null);

      // Callback to update devices list in real-time as they're discovered
      const onDeviceFound = (device: RovyDevice) => {
        setDevices((prevDevices) => {
          // Check if device already exists to avoid duplicates
          const exists = prevDevices.some((d) => d.id === device.id);
          if (exists) {
            // Update existing device (e.g., RSSI might have changed)
            return prevDevices.map((d) => (d.id === device.id ? device : d));
          }
          // Add new device
          return [...prevDevices, device];
        });
      };

      const foundDevices = await scanForRovy(onDeviceFound);

      // Final update with all devices (in case any were missed)
      setDevices(foundDevices);

      if (foundDevices.length === 0) {
        Alert.alert(
          "No Devices Found",
          "No devices were found. Make sure the robot is powered on and in BLE provisioning mode."
        );
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to scan for devices";
      Alert.alert("Scan Error", message);
    }
  }, [scanForRovy]);

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
    setIsOnSameNetwork(null);

    try {
      await refreshStatusRef.current();

      setTimeout(() => {
        setIsOnSameNetwork(true);
        setShowWifiConfig(false);
        setIsCheckingNetwork(false);
        isCheckingNetworkRef.current = false;
      }, 500);
    } catch (error) {
      console.log("Robot not on same network:", error);
      setIsOnSameNetwork(false);
      setIsCheckingNetwork(false);
      isCheckingNetworkRef.current = false;
    }
  }, []);

  /**
   * Step 2: Connect to selected device via Bluetooth
   */
  const handleConnect = useCallback(
    async (device: RovyDevice) => {
      try {
        setSelectedDevice(device);
        setShowWifiConfig(false);
        setIsOnSameNetwork(null);
        await connectToRovy(device.id);
        await refreshPhoneNetwork();
        await checkSameNetwork();
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to connect to device";
        Alert.alert("Connection Error", message);
        setSelectedDevice(null);
        setIsOnSameNetwork(null);
      }
    },
    [connectToRovy, refreshPhoneNetwork, checkSameNetwork]
  );

  /**
   * Handle "Connect" button - robot is on same network
   */
  const handleNetworkConnect = useCallback(async () => {
    try {
      setIsRobotConnecting(true);
      await refreshStatus();
      Alert.alert(
        "Connected",
        "Robot is connected on the same network. You can now control it."
      );
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to connect to robot";
      Alert.alert("Connection Error", message);
    } finally {
      setIsRobotConnecting(false);
    }
  }, [refreshStatus]);

  /**
   * Show Wi-Fi configuration form
   */
  const handleChangeWifi = useCallback(() => {
    setShowWifiConfig(true);
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
      await sendWifiConfig(ssid.trim(), password);
      setTimeout(async () => {
        if (wifiStatus === "connected") {
          await checkSameNetwork();
        }
      }, 2000);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Failed to send Wi-Fi configuration";
      Alert.alert("Configuration Error", message);
    }
  }, [ssid, password, sendWifiConfig, wifiStatus, checkSameNetwork]);

  /**
   * Disconnect from device
   */
  const handleDisconnect = useCallback(async () => {
    try {
      await disconnect();
      setSelectedDevice(null);
      setSsid("");
      setPassword("");
      setShowWifiConfig(false);
      setIsOnSameNetwork(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to disconnect";
      Alert.alert("Disconnect Error", message);
    }
  }, [disconnect]);

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

  const getStatusText = (): string => {
    switch (wifiStatus) {
      case "idle":
        return "Idle";
      case "connecting":
        return "Connecting...";
      case "connected":
        return "Connected";
      case "failed":
        return "Failed";
      default:
        return "Unknown";
    }
  };

  const getStatusColor = (): string => {
    switch (wifiStatus) {
      case "idle":
        return "#67686C";
      case "connecting":
        return "#FBBF24";
      case "connected":
        return "#1DD1A1";
      case "failed":
        return "#F87171";
      default:
        return "#67686C";
    }
  };

  const bluetoothStatus = useMemo(() => {
    if (error) {
      return { label: "Error", color: "#F87171" };
    }
    if (isConnecting) {
      return { label: "Connecting", color: "#FBBF24" };
    }
    if (isConnected) {
      return { label: "Connected", color: "#1DD1A1" };
    }
    return { label: "ON", color: "#5CC8FF" };
  }, [error, isConnecting, isConnected]);

  const wifiConnectionStatus = useMemo(() => {
    if (status?.network?.ip) {
      return { label: "Connected", color: "#1DD1A1" };
    }
    if (statusError) {
      return { label: "Offline", color: "#F87171" };
    }
    return { label: "Pending", color: "#FBBF24" };
  }, [status?.network?.ip, statusError]);

  const previousNetworks = useMemo(() => {
    const seen = new Set<string>();
    const candidates = [
      status?.network?.wifiSsid,
      status?.network?.ssid,
      status?.health?.network?.wifiSsid,
      status?.health?.network?.ssid,
    ];
    for (const candidate of candidates) {
      if (candidate && candidate.trim()) {
        seen.add(candidate.trim());
      }
    }
    return Array.from(seen);
  }, [status]);

  const overlayVisible =
    isConnecting ||
    isManualConnecting ||
    isRobotConnecting ||
    isCheckingNetwork ||
    isSendingConfig;

  const overlayMessage = (() => {
    if (isManualConnecting) {
      return "Establishing connection to your robot";
    }
    if (isRobotConnecting) {
      return "Refreshing robot status";
    }
    if (isSendingConfig) {
      return "Sending Wi-Fi credentials";
    }
    if (isCheckingNetwork) {
      return "Checking network connection";
    }
    return "Connecting to device";
  })();

  const wifiNetworkName =
    status?.network?.wifiSsid ||
    status?.network?.ssid ||
    status?.health?.network?.wifiSsid ||
    status?.health?.network?.ssid ||
    "Unknown network";

  const wifiIpAddress = status?.network?.ip || "Unavailable";
  const scanIconStyle = isScanning
    ? {
      transform: [{ rotate: scanRotation }],
    }
    : undefined;

  // Check if we're connected to robot (via BT, hotspot, or direct IP)
  const isConnectedToRobot = Boolean(
    isConnected || // Bluetooth connected
    status?.network?.ip || // Has network IP
    (baseUrl && baseUrl !== DEFAULT_ROBOT_BASE_URL) // Has custom base URL
  );

  // Scan robot Wi-Fi networks
  const handleScanRobotWifi = useCallback(async () => {
    if (!isConnectedToRobot || !api) {
      Alert.alert(
        "Not Connected",
        "Please connect to the robot first via Bluetooth or IP."
      );
      return;
    }

    setIsScanningRobotWifi(true);
    setRobotWifiNetworks([]);
    try {
      const response = await api.scanWifiNetworks();
      const networks = Array.isArray(response.networks)
        ? response.networks.map((n: any) => ({
          ssid: typeof n === "string" ? n : n.ssid || "",
          rssi: typeof n === "object" && n.rssi ? n.rssi : -100,
        }))
        : [];
      setRobotWifiNetworks(networks);
    } catch (error) {
      Alert.alert(
        "Scan Failed",
        error instanceof Error ? error.message : "Failed to scan Wi-Fi networks"
      );
    } finally {
      setIsScanningRobotWifi(false);
    }
  }, [isConnectedToRobot, api]);

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

          {error ? (
            <ThemedView style={styles.errorCard}>
              <ThemedText style={styles.errorText}>{error}</ThemedText>
            </ThemedView>
          ) : null}

          {statusError ? (
            <ThemedView style={styles.errorCard}>
              <ThemedText style={styles.errorText}>{statusError}</ThemedText>
            </ThemedView>
          ) : null}

          <View>
            <View style={styles.sectionHeader}>
              <View>
                <ThemedText style={styles.sectionTitle}>
                  Bluetooth
                </ThemedText>
              </View>
              <StatusPill
                color={bluetoothStatus.color}
                label={bluetoothStatus.label}
              />
            </View>

            <View style={styles.sectionCard}>
              <View style={styles.sectionHeader}>
                <View>
                  <ThemedText style={styles.sectionTitle}>
                    Nearby devices
                  </ThemedText>
                </View>
                <Pressable
                  style={[
                    styles.scanButton,
                    (isScanning || isConnecting) && styles.disabledPrimary,
                  ]}
                  onPress={handleScan}
                  disabled={isScanning || isConnecting}
                >
                  <Animated.View style={scanIconStyle}>
                    <IconSymbol
                      size={20}
                      name="arrow.trianglehead.2.clockwise"
                      color="#fff"
                    />
                  </Animated.View>
                </Pressable>
              </View>
              {isScanning ? (
                <View style={styles.inlineStatus}>
                  <ThemedText style={styles.statusLabelText}>
                    Scanning for devices...
                  </ThemedText>
                </View>
              ) : null}
              {devices.length === 0 && !isScanning && isConnected ? (
                <ThemedText style={styles.emptyStateText}>
                  Connected to a robot. Disconnect to scan again.
                </ThemedText>
              ) : (
                devices.length !== 0 && (
                  <View style={styles.deviceList}>
                    {devices.map((device) => {
                      const signalInfo = getSignalStrengthInfo(device.rssi);
                      return (
                        <Pressable
                          key={device.id}
                          style={[
                            styles.deviceItem,
                            selectedDevice?.id === device.id &&
                            styles.deviceSelected,
                          ]}
                          onPress={() => handleConnect(device)}
                          disabled={isConnecting}
                        >
                          <View style={styles.deviceHeader}>
                            {selectedDevice?.id === device.id && isConnected ? (
                              <View
                                style={[
                                  styles.signalDot,
                                  { backgroundColor: "#1DD1A1" },
                                ]}
                              />
                            ) : null}
                            <View>
                              <ThemedText style={styles.deviceName}>
                                {device.name || "JARVIS"}
                              </ThemedText>
                            </View>
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
                          </View>
                        </Pressable>
                      );
                    })}
                  </View>
                )
              )}
            </View>

            {isConnected ? (
              <Pressable
                style={[styles.secondaryButton, styles.outlineButton]}
                onPress={handleDisconnect}
                disabled={isSendingConfig}
              >
                <ThemedText style={styles.outlineButtonText}>
                  Disconnect
                </ThemedText>
              </Pressable>
            ) : null}
          </View>

          {/* Wi-Fi Section */}
          <View>
            <View style={styles.sectionHeader}>
              <View>
                <ThemedText style={styles.sectionTitle}>Wi-Fi</ThemedText>
              </View>
            </View>

            {/* Robot Wi-Fi Status Card */}
            <ThemedView style={styles.sectionCard}>
              <ThemedText style={styles.sectionTitle}>Robot Wi-Fi</ThemedText>
              <View style={styles.statusRow}>
                <ThemedText style={styles.statusLabel}>Status:</ThemedText>
                <ThemedText style={styles.statusValue}>
                  {status?.network?.wifiSsid || status?.network?.ssid
                    ? `Connected to ${status.network.wifiSsid || status.network.ssid
                    }`
                    : "Not connected"}
                </ThemedText>
              </View>

              {isConnectedToRobot &&
                !status?.network?.wifiSsid &&
                !status?.network?.ssid && (
                  <>
                    <Pressable
                      style={[
                        styles.primaryButton,
                        (isScanningRobotWifi || !isConnectedToRobot) &&
                        styles.disabledPrimary,
                      ]}
                      onPress={handleScanRobotWifi}
                      disabled={isScanningRobotWifi || !isConnectedToRobot}
                    >
                      {isScanningRobotWifi ? (
                        <ActivityIndicator color="#04110B" />
                      ) : (
                        <ThemedText style={styles.primaryButtonText}>
                          Scan networks
                        </ThemedText>
                      )}
                    </Pressable>

                    {robotWifiNetworks.length > 0 && (
                      <View style={styles.wifiList}>
                        <ThemedText style={styles.subsectionTitle}>
                          Available networks
                        </ThemedText>
                        {robotWifiNetworks.map((network, index) => {
                          const signalInfo = getSignalStrengthInfo(
                            network.rssi
                          );
                          return (
                            <Pressable
                              key={`${network.ssid}-${index}`}
                              style={styles.wifiItem}
                              onPress={() => {
                                // TODO: Open password input modal
                                Alert.alert(
                                  "Connect",
                                  `Connect to ${network.ssid}? (Password input coming soon)`
                                );
                              }}
                            >
                              <ThemedText style={styles.wifiSsid}>
                                {network.ssid}
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
                  </>
                )}
            </ThemedView>

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
