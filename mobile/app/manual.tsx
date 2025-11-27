import { Image } from 'expo-image';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
     ActivityIndicator,
     Pressable,
     StyleSheet,
     Text,
     View
} from 'react-native';

import { useRouter } from 'expo-router';

import { CameraVideo } from '@/components/camera-video';
import { Joystick } from '@/components/joystick';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { useRobot } from '@/context/robot-provider';
import { SafeAreaView } from 'react-native-safe-area-context';
import { cmd_lights_ctrl, cmdJsonCmd } from '@/services/json-socket';

export default function ManualScreen() {
     const { api, baseUrl, } = useRobot();
     const router = useRouter();
     const [joystick, setJoystick] = useState({ l: 0, r: 0 });
     const [error, setError] = useState<string | null>(null);
     const [isCapturing, setIsCapturing] = useState(false);
     const [lastSnapshot, setLastSnapshot] = useState<string | null>(null);
     const [currentFrame, setCurrentFrame] = useState<string | null>(null);
     const [isStreaming, setIsStreaming] = useState(false);
     const [isConnecting, setIsConnecting] = useState(false);
     const [isLightOn, setIsLightOn] = useState(false)
     const wsRef = useRef<WebSocket | null>(null);

     // WebSocket URL
     const wsUrl = useMemo(() => {
          if (!baseUrl) return undefined;

          try {
               // Normalize URL (add scheme if missing)
               const normalizedUrl = baseUrl.startsWith('http')
                    ? baseUrl
                    : `http://${baseUrl}`; // default to http for IPs or unknown

               const parsedUrl = new URL(normalizedUrl);

               const host = parsedUrl.hostname;

               // Detect if hostname is IP
               const isIp =
                    /^\d{1,3}(\.\d{1,3}){3}$/.test(host) ||
                    host === "localhost";

               // Set protocol
               if (isIp) {
                    parsedUrl.protocol = "ws:";   // <-- IP → always ws
               } else {
                    parsedUrl.protocol = parsedUrl.protocol === "https:" ? "wss:" : "ws:";
               }

               // Set the WS path
               parsedUrl.pathname =
                    `${parsedUrl.pathname.replace(/\/$/, "")}/camera/ws`;

               parsedUrl.search = "";

               return parsedUrl.toString();
          } catch (err) {
               console.warn("Invalid base URL for WebSocket", err);
               return undefined;
          }
     }, [baseUrl]);



     const connectWebSocket = useCallback(() => {
          if (!wsUrl) {
               setError('No WebSocket URL available');
               return;
          }

          setIsConnecting(true);
          setError(null);

          console.log('Connecting to WebSocket:', wsUrl);

          const ws = new WebSocket(wsUrl);
          wsRef.current = ws;

          ws.onopen = () => {
               console.log('WebSocket connected');
               setIsConnecting(false);
               setIsStreaming(true);
               setError(null);
          };

          ws.onmessage = (event) => {
               try {
                    const data = JSON.parse(event.data);

                    if (data.error) {
                         console.error('Stream error:', data.error);
                         setError(data.error);
                         return;
                    }

                    if (data.frame) {
                         // Update frame with base64 data
                         setCurrentFrame(`data:image/jpeg;base64,${data.frame}`);
                    }
               } catch (err) {
                    console.error('Error parsing WebSocket message:', err);
               }
          };

          ws.onerror = (event) => {
               console.error('WebSocket error:', event);
               setError('WebSocket connection error');
               setIsConnecting(false);
               setIsStreaming(false);
          };

          ws.onclose = (event) => {
               console.log('WebSocket closed:', event.code, event.reason);
               setIsStreaming(false);
               setIsConnecting(false);

               if (!event.wasClean) {
                    setError(`Connection closed unexpectedly (${event.code})`);
               }
          };
     }, [wsUrl]);

     const disconnectWebSocket = useCallback(() => {
          if (wsRef.current) {
               wsRef.current.close();
               wsRef.current = null;
          }
          setIsStreaming(false);
          setIsConnecting(false);
          setCurrentFrame(null);
     }, []);

     // Auto-start streaming when wsUrl becomes available
     useEffect(() => {
          if (wsUrl && !isStreaming && !isConnecting) {
               console.log('Auto-starting stream...');
               connectWebSocket();
          }
     }, [wsUrl, isStreaming, isConnecting, connectWebSocket]);

     // Cleanup on unmount
     useEffect(() => {
          return () => {
               disconnectWebSocket();
          };
     }, [disconnectWebSocket]);

     const handleToggleStream = useCallback(() => {
          if (isStreaming || isConnecting) {
               disconnectWebSocket();
          } else {
               connectWebSocket();
          }
     }, [isStreaming, isConnecting, connectWebSocket, disconnectWebSocket]);

     const resolveSnapshotUrl = useCallback(() => {
          if (!baseUrl) {
               return null;
          }
          const cacheBuster = Date.now();
          return `${api.snapshotUrl}?ts=${cacheBuster}`;
     }, [api, baseUrl]);

     const handleSnapshot = useCallback(async () => {
          setIsCapturing(true);
          try {
               const metadata = await api.capturePhoto();
               const url =
                    (metadata?.url as string | undefined) ||
                    (metadata?.snapshotUrl as string | undefined) ||
                    (metadata?.imageUrl as string | undefined) ||
                    (metadata?.path as string | undefined) ||
                    resolveSnapshotUrl();
               setLastSnapshot(url ?? null);
          } catch (error) {
               console.warn('Snapshot failed', error);
               setLastSnapshot(resolveSnapshotUrl());
          } finally {
               setIsCapturing(false);
          }
     }, [api, resolveSnapshotUrl]);


     const handleLightCTL = useCallback(
          () => {
               cmdJsonCmd({ T: cmd_lights_ctrl, IO4: isLightOn ? 0 : 115, IO5: isLightOn ? 0 : 115 }, baseUrl);
               setIsLightOn(!isLightOn)
          },
          [baseUrl, isLightOn],
     );

     return (
          <SafeAreaView style={styles.safeArea} edges={["top", "bottom"]}>
               <ThemedView style={styles.container}>
                    <View style={styles.headerRow}>
                         <Pressable style={styles.backButton} onPress={() => router.back()}>
                              <IconSymbol name="chevron.left" size={16} color="#E5E7EB" />
                         </Pressable>
                         <ThemedText type="title">Manual control</ThemedText>
                    </View>

                    {/* {wsUrl && (
                         <View style={styles.statusBar}>
                              <ThemedText style={styles.statusText}>
                                   {isConnecting ? 'Connecting...' :
                                        isStreaming ? `Streaming | ${frameCount} frames | ${fps.toFixed(1)} fps` :
                                             'Disconnected'}
                              </ThemedText>
                              <Pressable
                                   style={[
                                        styles.streamButton,
                                        isStreaming && styles.streamButtonActive,
                                        isConnecting && styles.streamButtonConnecting
                                   ]}
                                   onPress={handleToggleStream}
                                   disabled={isConnecting}
                              >
                                   {isConnecting ? (
                                        <ActivityIndicator size="small" color="#04110B" />
                                   ) : (
                                        <ThemedText style={styles.streamButtonText}>
                                             {isStreaming ? '⏸ Stop' : '▶ Start'}
                                        </ThemedText>
                                   )}
                              </Pressable>
                         </View>
                    )} */}

                    <View style={styles.videoFeed}>
                         <View style={{ position: "relative" }}>
                              <View style={{ position: "absolute", zIndex: 2 }}>
                                   <View style={{ flex: 1, alignItems: "flex-end", justifyContent: "flex-start" }}>
                                        <Pressable style={styles.feedLight} onPress={handleLightCTL}>
                                             <IconSymbol name='bolt' size={20} color="#1DD1A1" />
                                             <Text style={styles.feedLightText}>
                                                  {isLightOn ? "ON" : "OFF"}
                                             </Text>
                                        </Pressable>
                                   </View>
                              </View>
                         </View>
                         <CameraVideo
                              wsUrl={wsUrl}
                              currentFrame={currentFrame}
                              isConnecting={isConnecting}
                              isStreaming={isStreaming}
                              error={error}
                              onToggleStream={handleToggleStream}
                         />
                    </View>

                    <View style={styles.row}>
                         <Pressable
                              style={styles.secondaryButton}
                              onPress={handleToggleStream}
                         >
                              <ThemedText>
                                   {isStreaming ? 'Reconnect' : 'Connect'}
                              </ThemedText>
                         </Pressable>
                         <Pressable style={styles.primaryButton} onPress={handleSnapshot}>
                              {isCapturing ? (
                                   <ActivityIndicator color="#04110B" />
                              ) : (
                                   <ThemedText style={styles.primaryText}>Capture photo</ThemedText>
                              )}
                         </Pressable>
                    </View>

                    {lastSnapshot ? (
                         <ThemedView style={styles.snapshotCard}>
                              <ThemedText type="subtitle">Latest snapshot</ThemedText>
                              <Image
                                   source={{ uri: lastSnapshot }}
                                   style={styles.snapshot}
                                   contentFit="cover"
                              />
                         </ThemedView>
                    ) : null}

                    <ThemedView style={styles.joystickCard}>
                         <Joystick onChange={setJoystick} />
                         <ThemedText style={styles.joystickValue}>
                              L: {joystick.l.toFixed(2)} R: {joystick.r.toFixed(2)}
                         </ThemedText>
                    </ThemedView>
               </ThemedView>
          </SafeAreaView>
     );
}

const styles = StyleSheet.create({
     safeArea: {
          flex: 1,
          backgroundColor: "#161616",
     },
     container: {
          flex: 1,
          padding: 24,
          gap: 16,
          backgroundColor: '#161616',
     },
     headerRow: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: 8
     },
     backButton: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: 6,
          padding: 8,
          borderWidth: 1,
          borderColor: '#202020',
          backgroundColor: '#1C1C1C',
     },
     backButtonText: {
          color: '#E5E7EB',
     },
     feedLight: {
          borderWidth: 1,
          borderColor: "#1DD1A1",
          paddingInline: 6,
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          marginVertical: 5
     },
     feedLightText: {
          color: "#1DD1A1",
          fontSize: 14
     },
     row: {
          flexDirection: 'row',
          gap: 16,
          alignItems: 'center',
     },
     videoFeed: {
          position: "relative",
          zIndex: 0
     },
     primaryButton: {
          flex: 1,
          backgroundColor: '#1DD1A1',
          borderRadius: 0,
          paddingVertical: 16,
          alignItems: 'center',
     },
     primaryText: {
          color: '#04110B',
     },
     secondaryButton: {
          flex: 1,
          borderRadius: 0,
          paddingVertical: 16,
          alignItems: 'center',
          borderWidth: 1,
          borderColor: '#202020',
          backgroundColor: '#1B1B1B',
     },
     snapshotCard: {
          gap: 16,
          padding: 20,
          borderRadius: 0,
          borderWidth: 1,
          borderColor: '#202020',
          backgroundColor: '#1C1C1C',
     },
     snapshot: {
          width: '100%',
          aspectRatio: 4 / 3,
          borderRadius: 0,
     },
     joystickCard: {
          gap: 16,
          padding: 20,
          borderRadius: 0,
          backgroundColor: '#161616',
          alignItems: 'center',
     },
     joystickValue: {
          fontVariant: ['tabular-nums'],
          color: '#E5E7EB',
     },
});