import React, { useMemo } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Animated, { FadeInDown } from 'react-native-reanimated';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { IconSymbol } from '@/components/ui/icon-symbol';

interface Summary {
  id: string;
  title: string;
  type: 'meeting' | 'lecture' | 'conversation' | 'note';
  content: string;
  date: Date;
  icon: string;
}

export default function SummariesScreen() {

  const summaries: Summary[] = useMemo(() => [
    {
      id: '1',
      title: 'Meeting Summary 1',
      type: 'meeting',
      content: 'Discussed project timeline and deliverables. Key decisions: launch date set for Q2, team expansion planned for next month.',
      date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
      icon: 'person.2.fill',
    },
    {
      id: '2',
      title: 'Lecture Summary - AI Fundamentals',
      type: 'lecture',
      content: 'Covered neural networks, backpropagation, and gradient descent. Important concepts: activation functions, loss functions, and optimization algorithms.',
      date: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), // 5 days ago
      icon: 'book.fill',
    },
    {
      id: '3',
      title: 'Meeting Summary 2',
      type: 'meeting',
      content: 'Weekly standup: reviewed sprint progress, identified blockers, and planned next week\'s tasks. All team members present.',
      date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
      icon: 'person.2.fill',
    },
    {
      id: '4',
      title: 'Lecture Summary - Machine Learning',
      type: 'lecture',
      content: 'Introduction to supervised learning, classification vs regression, and evaluation metrics. Hands-on exercise with scikit-learn.',
      date: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000), // 10 days ago
      icon: 'book.fill',
    },
    {
      id: '5',
      title: 'Conversation Summary',
      type: 'conversation',
      content: 'User discussed preferences for robot behavior and voice settings. Updated configuration based on feedback.',
      date: new Date(Date.now() - 12 * 24 * 60 * 60 * 1000), // 12 days ago
      icon: 'message.fill',
    },
    {
      id: '6',
      title: 'Quick Note',
      type: 'note',
      content: 'Reminder: Check battery levels regularly. Scheduled maintenance for next week.',
      date: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000), // 14 days ago
      icon: 'note.text',
    },
  ], []);

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <ScrollView 
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        <ThemedView style={styles.container}>
          <Animated.View entering={FadeInDown.duration(400)}>
            <ThemedText type="title">Summaries</ThemedText>
            <ThemedText style={styles.description}>
              Meeting notes, lecture summaries, and conversation logs
            </ThemedText>
          </Animated.View>

          {/* Summaries */}
          <Animated.View 
            entering={FadeInDown.delay(100).duration(400)}
            style={styles.section}
          >
            <ThemedText style={styles.sectionTitle}>SUMMARIES</ThemedText>
            <View style={styles.summariesList}>
              {summaries.map((summary, index) => (
                <Animated.View 
                  key={summary.id}
                  entering={FadeInDown.delay(150 + index * 50).duration(300)}
                >
                  <ThemedView style={styles.summaryCard}>
                    <View style={styles.summaryHeader}>
                      <View style={styles.summaryIconContainer}>
                        <IconSymbol name={summary.icon} size={20} color="#1DD1A1" />
                  </View>
                      <View style={styles.summaryTitleContainer}>
                        <ThemedText style={styles.summaryTitle}>{summary.title}</ThemedText>
                        <ThemedText style={styles.summaryDate}>
                          {summary.date.toLocaleDateString('en-US', { 
                            month: 'short', 
                            day: 'numeric',
                            year: 'numeric'
                          })}
                        </ThemedText>
                  </View>
            </View>
                    <ThemedText style={styles.summaryContent}>{summary.content}</ThemedText>
            </ThemedView>
                </Animated.View>
              ))}
            </View>
          </Animated.View>
        </ThemedView>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0F0F0F',
  },
  scroll: {
    flexGrow: 1,
    paddingBottom: 40,
  },
  container: {
    flex: 1,
    padding: 20,
    gap: 24,
    backgroundColor: '#0F0F0F',
  },
  description: {
    color: '#9CA3AF',
    marginTop: 6,
    fontSize: 14,
  },
  section: {
    gap: 12,
  },
  sectionTitle: {
    fontSize: 12,
    fontFamily: 'JetBrainsMono_600SemiBold',
    color: '#67686C',
    textTransform: 'uppercase',
    letterSpacing: 1.2,
  },
  summariesList: {
    gap: 12,
  },
  summaryCard: {
    padding: 16,
    backgroundColor: 'rgba(26, 26, 26, 0.7)',
    borderWidth: 1,
    borderColor: 'rgba(37, 37, 37, 0.6)',
    borderRadius: 12,
    gap: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 2,
  },
  summaryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  summaryIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 10,
    backgroundColor: 'rgba(29, 209, 161, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  summaryTitleContainer: {
    flex: 1,
  },
  summaryTitle: {
    fontSize: 15,
    fontFamily: 'JetBrainsMono_600SemiBold',
    color: '#E5E7EB',
    marginBottom: 2,
  },
  summaryDate: {
    fontSize: 12,
    color: '#67686C',
  },
  summaryContent: {
    fontSize: 14,
    color: '#9CA3AF',
    lineHeight: 20,
  },
});
