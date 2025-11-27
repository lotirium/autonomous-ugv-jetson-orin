import React from 'react';
import { ScrollView, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';

export default function MemoryScreen() {
  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <ThemedView style={styles.container}>
          <ThemedText type="title">Memory</ThemedText>
          <ThemedText style={styles.description}>
            The robot keeps track of preferences, routines, and contextual cues to provide a more personal experience.
          </ThemedText>

          <ThemedView style={styles.card}>
            <ThemedText type="subtitle">No memories yet</ThemedText>
            <ThemedText>
              Interact with your robot to start building personalized memories and preferences.
            </ThemedText>
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
});
