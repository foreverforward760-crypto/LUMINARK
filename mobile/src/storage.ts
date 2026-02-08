/**
 * LUMINARK Inner Child Journal - Persistent Storage
 * Uses AsyncStorage for local persistence on device
 */

import AsyncStorage from "@react-native-async-storage/async-storage";

export type JournalEntry = {
  id: string;
  created_at: string;
  title: string;
  tags: string[];
  content: string;
};

const JOURNAL_KEY = "luminark_inner_child_journal_v1";

/**
 * Load all journal entries from local storage
 */
export async function loadJournal(): Promise<JournalEntry[]> {
  try {
    const raw = await AsyncStorage.getItem(JOURNAL_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as JournalEntry[]) : [];
  } catch (error) {
    console.error("Failed to load journal:", error);
    return [];
  }
}

/**
 * Save all journal entries to local storage
 */
export async function saveJournal(entries: JournalEntry[]): Promise<void> {
  try {
    await AsyncStorage.setItem(JOURNAL_KEY, JSON.stringify(entries));
  } catch (error) {
    console.error("Failed to save journal:", error);
    throw error;
  }
}

/**
 * Add a new entry to the journal
 */
export async function addEntry(entry: JournalEntry): Promise<JournalEntry[]> {
  const current = await loadJournal();
  const next = [...current, entry];
  await saveJournal(next);
  return next;
}

/**
 * Update an existing entry
 */
export async function updateEntry(id: string, updates: Partial<JournalEntry>): Promise<JournalEntry[]> {
  const current = await loadJournal();
  const next = current.map((e) => (e.id === id ? { ...e, ...updates } : e));
  await saveJournal(next);
  return next;
}

/**
 * Delete an entry by ID
 */
export async function deleteEntry(id: string): Promise<JournalEntry[]> {
  const current = await loadJournal();
  const next = current.filter((e) => e.id !== id);
  await saveJournal(next);
  return next;
}

/**
 * Clear all journal entries
 */
export async function clearJournal(): Promise<void> {
  await AsyncStorage.removeItem(JOURNAL_KEY);
}

/**
 * Generate a unique entry ID
 */
export function generateEntryId(): string {
  return `entry_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
