import React from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';

const memoryData = {
  summary:
    'The robot keeps track of preferences, routines, and contextual cues to provide a more personal experience.',
  lastSync: 'Updated 5 minutes ago',
  categories: [
    {
      title: 'Personal preferences',
      highlights: [
        { label: 'Favorite music', value: 'Morning jazz playlists', confidence: 'High' },
        { label: 'Coffee order', value: 'Oat milk latte, medium roast', confidence: 'Medium' },
        { label: 'Lighting preference', value: 'Warm ambient lights after 8 PM', confidence: 'High' },
      ],
    },
    {
      title: 'Daily routines',
      highlights: [
        { label: 'Wake-up routine', value: 'Launches gentle alarm and opens blinds at 7:00 AM', confidence: 'High' },
        { label: 'Workout reminder', value: 'Ping at 6:30 PM on weekdays for stretching', confidence: 'Medium' },
        { label: 'Quiet hours', value: 'Reduce notifications after 10:30 PM', confidence: 'High' },
      ],
    },
    {
      title: 'Recent interactions',
      highlights: [
        { label: 'Last request', value: 'Create grocery list for Mediterranean dinner', confidence: 'High' },
        { label: 'Follow-up needed', value: 'Confirm reservation at favorite restaurant', confidence: 'Low' },
        { label: 'Mood observations', value: 'Noted upbeat tone during afternoon check-in', confidence: 'Medium' },
      ],
    },
  ],
  insights: [
    {
      title: 'Opportunities',
      description:
        'Suggest a new jazz playlist for the morning routine and confirm if the stretching reminder is still helpful.',
    },
    {
      title: 'Questions to ask',
      description: 'Would you like me to adjust the quiet hours while guests are visiting this weekend?',
    },
  ],
};

export default function MemoryScreen() {
  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <ThemedView style={styles.container}>
          <ThemedText type="title">Memory</ThemedText>
          <ThemedText style={styles.description}>{memoryData.summary}</ThemedText>

          <ThemedView style={styles.card}>
            <View style={styles.rowBetween}>
              <ThemedText type="subtitle">Sync status</ThemedText>
              <ThemedText style={styles.meta}>{memoryData.lastSync}</ThemedText>
            </View>
            <ThemedText>
              Memory is stored securely on-device and refreshed whenever you interact with the robot.
            </ThemedText>
          </ThemedView>

          {memoryData.categories.map((category) => (
            <ThemedView key={category.title} style={styles.card}>
              <ThemedText type="subtitle">{category.title}</ThemedText>
              {category.highlights.map((item) => (
                <View key={item.label} style={styles.entry}>
                  <View style={styles.entryText}>
                    <ThemedText style={styles.entryLabel}>{item.label}</ThemedText>
                    <ThemedText>{item.value}</ThemedText>
                  </View>
                  <ThemedText style={styles.confidence}>Confidence: {item.confidence}</ThemedText>
                </View>
              ))}
            </ThemedView>
          ))}

          <ThemedView style={styles.card}>
            <ThemedText type="subtitle">Insights</ThemedText>
            {memoryData.insights.map((insight) => (
              <View key={insight.title} style={styles.insight}>
                <ThemedText style={styles.entryLabel}>{insight.title}</ThemedText>
                <ThemedText>{insight.description}</ThemedText>
              </View>
            ))}
          </ThemedView>
        </ThemedView>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  scroll: {
    flexGrow: 1,
  },
  container: {
    flex: 1,
    padding: 20,
    gap: 16,
  },
  description: {
    opacity: 0.8,
  },
  card: {
    gap: 12,
    padding: 16,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 0,
    borderColor: '#202020',
    backgroundColor: '#1C1C1C',
  },
  rowBetween: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  meta: {
    opacity: 0.7,
  },
  entry: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 16,
  },
  entryText: {
    flex: 1,
    gap: 4,
  },
  entryLabel: {
    fontWeight: '600',
  },
  confidence: {
    alignSelf: 'flex-start',
    opacity: 0.7,
    fontSize: 12,
  },
  insight: {
    gap: 4,
  },
});
