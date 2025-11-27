import { Image } from 'expo-image';
import { Link, useRouter } from 'expo-router';
import React from 'react';
import { Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { useRobot } from '@/context/robot-provider';

const BEHAVIORS = [
  {
    id: 'hello_world',
    name: 'hello_world',
    tier: 'Innate',
    description: 'Loads at startup',
    status: 'Active',
    statusColor: '#34D399',
  },
  {
    id: 'default_directive',
    name: 'default_directive',
    tier: 'Innate',
    description: 'Fallback instructions for navigation',
    status: 'Idle',
    statusColor: '#FCD34D',
  },
  {
    id: 'tools_giving_directive',
    name: 'tools_giving_directive',
    tier: 'Agentic',
    description: 'Enables remote toolchains',
    status: 'Standby',
    statusColor: '#A5B4FC',
  },
] as const;

const CONTROL_MODES = [
  {
    id: 'manual',
    label: 'Manual',
    href: '/manual' as const,
    cardIcon: 'camera.fill' as const,
    cardIconColor: '#1DD1A1',
    chevronColor: '#94A3B8',
    cardVariant: 'camera' as const,
    modeIcon: 'bolt.fill' as const,
  },
  {
    id: 'agentic',
    label: 'Agentic',
    href: '/agentic' as const,
    cardIcon: 'mic.fill' as const,
    cardIconColor: '#0F1512',
    chevronColor: '#0F1512',
    cardVariant: 'voice' as const,
    modeIcon: 'sparkles' as const,
  },
] as const;

export default function HomeScreen() {
  const { status } = useRobot();
  const router = useRouter();
  const batteryRaw = status?.battery ?? status?.telemetry?.battery ?? status?.health?.battery;
  const batteryLevel = typeof batteryRaw === 'number' ? Math.round(batteryRaw) : undefined;
  const batteryLabel = batteryLevel !== undefined ? `${batteryLevel}%` : '—';
  const healthLabel = batteryLevel === undefined
    ? 'Unknown'
    : batteryLevel >= 60
      ? 'Good'
      : batteryLevel >= 30
        ? 'Low'
        : 'Critical';

  const wifiLabel = status?.network?.network_name ?? status?.network?.wifiSsid ?? status?.network?.ssid ?? 'Offline';
  const ipAddress = status?.network?.ip;
  const ipLabel = ipAddress ? `IP ${ipAddress}` : 'Awaiting link';

  const mode = status?.mode?.toLowerCase();
  const isAgentic = mode === 'agentic';

  const stats = [
    {
      id: 'battery',
      label: 'Battery',
      value: batteryLabel,
      caption: healthLabel,
      icon: 'battery.75' as const,
      iconColor: '#34D399',
    },
    {
      id: 'network',
      label: 'Network',
      value: wifiLabel,
      caption: ipLabel,
      icon: 'antenna.radiowaves.left.and.right' as const,
      iconColor: '#60A5FA',
    },
  ];

  return (
    <SafeAreaView style={styles.safeArea} edges={["top"]}>
      <ThemedView style={styles.screen}>
        <ScrollView
          contentContainerStyle={styles.content}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.headerRow}>
            <View>
              <ThemedText type="title" style={styles.missionLabel}>
                JARVIS
              </ThemedText>
              <View style={styles.headerStatus}>
                <IconSymbol name="battery.75" color="#34D399" size={20} />
                <ThemedText style={styles.headerStatusText}>
                  {batteryLabel}
                </ThemedText>
                <View style={styles.headerDot} />
                <ThemedText style={styles.headerStatusMeta}>
                  {healthLabel}
                </ThemedText>
              </View>
            </View>
          </View>

          <Image
            source={require("@/assets/images/rovy.png")}
            style={styles.robotImage}
            contentFit="cover"
          />

          <View style={styles.statGrid}>
            {stats.map((stat) => (
              <View key={stat.id} style={styles.statCard}>
                <View style={styles.statIconWrapper}>
                  <IconSymbol
                    name={stat.icon}
                    size={18}
                    color={stat.iconColor}
                  />
                </View>
                <ThemedText style={styles.statLabel}>{stat.label}</ThemedText>
                <ThemedText style={styles.statValue}>{stat.value}</ThemedText>
                <ThemedText style={styles.statCaption}>
                  {stat.caption}
                </ThemedText>
              </View>
            ))}
          </View>

          <View style={styles.modeRow}>
            {CONTROL_MODES.map((control) => {
              const isActive =
                control.id === "agentic" ? isAgentic : !isAgentic;
              return (
                <Pressable
                  key={control.id}
                  style={({ pressed }) => [
                    styles.modeButton,
                    isActive && styles.modeButtonActive,
                    pressed && styles.modeButtonPressed,
                  ]}
                  onPress={() => router.push(control.href)}
                >
                  <IconSymbol
                    name={control.modeIcon}
                    size={18}
                    color={isActive ? "#161616" : "#CBD5F5"}
                  />
                  <ThemedText
                    style={[
                      styles.modeButtonText,
                      isActive && styles.modeButtonTextActive,
                    ]}
                  >
                    {control.label}
                  </ThemedText>
                </Pressable>
              );
            })}
          </View>

          <View style={styles.behaviorHeader}>
            <ThemedText type="subtitle" style={styles.behaviorTitle}>
              Behaviors
            </ThemedText>
            <ThemedText type="link">View logs</ThemedText>
          </View>

          <View style={styles.behaviorList}>
            {BEHAVIORS.map((behavior) => (
              <Pressable
                key={behavior.id}
                style={({ pressed }) => [
                  styles.behaviorCard,
                  pressed && styles.behaviorCardPressed,
                ]}
              >
                <View style={styles.behaviorTextGroup}>
                  <View style={styles.behaviorMetaRow}>
                    <View style={styles.behaviorBadge}>
                      <ThemedText style={styles.behaviorBadgeText}>
                        {behavior.tier}
                      </ThemedText>
                    </View>
                    <View style={styles.behaviorStatusPill}>
                      <IconSymbol
                        name="checkmark.circle.fill"
                        size={14}
                        color={behavior.statusColor}
                      />
                      <ThemedText style={styles.behaviorStatusText}>
                        {behavior.status}
                      </ThemedText>
                    </View>
                  </View>
                  <ThemedText style={styles.behaviorName}>
                    {behavior.name}
                  </ThemedText>
                  <ThemedText style={styles.behaviorDescription}>
                    {behavior.description}
                  </ThemedText>
                </View>
                <IconSymbol name="chevron.right" size={20} color="#94A3B8" />
              </Pressable>
            ))}
          </View>
        </ScrollView>
      </ThemedView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#161616',
  },
  screen: {
    flex: 1,
    backgroundColor: '#161616',
  },
  content: {
    padding: 24,
    paddingBottom: 48,
    gap: 24,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  missionLabel: {
    color: '#F9FAFB',
  },
  missionSubtitle: {
    color: '#67686C',
    marginTop: 4,
  },
  headerStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 6,
  },
  headerStatusText: {
    fontFamily: 'JetBrainsMono_600SemiBold',
    color: '#F9FAFB',
  },
  headerStatusMeta: {
    color: '#67686C',
    fontSize: 14,
  },
  headerDot: {
    width: 4,
    height: 4,
    backgroundColor: '#1C1C1C',
  },
  robotImage: {
    width: '100%',
    aspectRatio: 4 / 3,
  },
  statGrid: {
    flexDirection: 'row',
    gap: 12,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#1C1C1C',
    padding: 16,
    gap: 6,
    borderWidth: 1,
    borderColor: '#202020',
  },
  statIconWrapper: {
    width: 32,
    height: 32,
    backgroundColor: '#202020',
    alignItems: 'center',
    justifyContent: 'center',
  },
  statLabel: {
    color: '#67686C',
    fontSize: 14,
  },
  statValue: {
    fontFamily: 'JetBrainsMono_600SemiBold',
    fontSize: 20,
    color: '#F9FAFB',
  },
  statCaption: {
    color: '#6B7280',
    fontSize: 14,
  },
  modeRow: {
    flexDirection: 'row',
    gap: 12,
  },
  modeButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    backgroundColor: '#1C1C1C',
    borderWidth: 1,
    borderColor: '#202020',
  },
  modeButtonActive: {
    backgroundColor: '#F8FAFC',
    borderColor: '#F8FAFC',
  },
  modeButtonPressed: {
    opacity: 0.85,
  },
  modeButtonText: {
    color: '#E5E7EB',
    fontFamily: 'JetBrainsMono_600SemiBold',
  },
  modeButtonTextActive: {
    color: '#161616',
  },
  behaviorHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  behaviorTitle: {
    color: '#F9FAFB',
  },
  behaviorList: {
    gap: 12,
  },
  behaviorCard: {
    backgroundColor: '#1C1C1C',
    padding: 16,
    borderWidth: 1,
    borderColor: '#202020',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 16,
  },
  behaviorCardPressed: {
    borderColor: '#1E3A8A',
  },
  behaviorTextGroup: {
    flex: 1,
    gap: 8,
  },
  behaviorMetaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  behaviorBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: '#202020',
  },
  behaviorBadgeText: {
    fontSize: 12,
    color: '#67686C',
    fontFamily: 'JetBrainsMono_600SemiBold',
  },
  behaviorStatusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: '#202020',
  },
  behaviorStatusText: {
    color: '#D1D5DB',
    fontSize: 12,
    fontFamily: 'JetBrainsMono_400Regular',
  },
  behaviorName: {
    color: '#F9FAFB',
    fontFamily: 'JetBrainsMono_600SemiBold',
    fontSize: 16,
  },
  behaviorDescription: {
    color: '#67686C',
    fontSize: 14,
  },
  cameraButton: {
    backgroundColor: '#1C1C1C',
    padding: 16,
    borderWidth: 1,
    borderColor: '#202020',
  },
  cameraButtonPressed: {
    borderColor: '#1DD1A1',
    opacity: 0.9,
  },
  cameraButtonContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  cameraButtonText: {
    flex: 1,
    color: '#F9FAFB',
    fontFamily: 'JetBrainsMono_600SemiBold',
    fontSize: 16,
  },
  voiceButton: {
    backgroundColor: '#1DD1A1',
    padding: 16,
    borderWidth: 1,
    borderColor: '#1DD1A1',
  },
  voiceButtonPressed: {
    backgroundColor: '#0DAA80',
    borderColor: '#0DAA80',
  },
  voiceButtonContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  voiceButtonText: {
    flex: 1,
    color: '#04110B',
    fontFamily: 'JetBrainsMono_600SemiBold',
    fontSize: 16,
  },
});
