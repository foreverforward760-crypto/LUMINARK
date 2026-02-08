/**
 * LUMINARK Inner Child Journal Screen
 * Main journal interface with write, browse, and export functionality
 */

import React, { useEffect, useState } from "react";
import { View, Text, ScrollView, StyleSheet, Alert } from "react-native";
import { Btn, Card, H2, Input, Label, colors } from "../ui";
import type { JournalEntry } from "../storage";
import { addEntry, loadJournal, deleteEntry, generateEntryId } from "../storage";
import { shareJournalCsv, shareJournalPdf, isSharingAvailable } from "../export";

interface JournalProps {
  back?: () => void;
}

export function Journal({ back }: JournalProps) {
  const [title, setTitle] = useState("");
  const [tags, setTags] = useState("");
  const [content, setContent] = useState("");
  const [entries, setEntries] = useState<JournalEntry[]>([]);
  const [status, setStatus] = useState("");
  const [canShare, setCanShare] = useState(false);

  useEffect(() => {
    (async () => {
      const loaded = await loadJournal();
      setEntries(loaded);
      setCanShare(await isSharingAvailable());
    })();
  }, []);

  function showStatus(msg: string, duration = 1500) {
    setStatus(msg);
    setTimeout(() => setStatus(""), duration);
  }

  async function onSave() {
    if (!content.trim()) return;

    showStatus("Saving...");
    try {
      const entry: JournalEntry = {
        id: generateEntryId(),
        created_at: new Date().toISOString(),
        title: title.trim() || "Untitled",
        tags: tags
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
        content: content.trim(),
      };

      const next = await addEntry(entry);
      setEntries(next);
      setTitle("");
      setTags("");
      setContent("");
      showStatus("Saved");
    } catch (error) {
      showStatus("Failed to save");
    }
  }

  async function onDelete(id: string) {
    Alert.alert("Delete Entry", "Are you sure you want to delete this entry?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: async () => {
          showStatus("Deleting...");
          try {
            const next = await deleteEntry(id);
            setEntries(next);
            showStatus("Deleted");
          } catch (error) {
            showStatus("Failed to delete");
          }
        },
      },
    ]);
  }

  async function onExportCsv() {
    if (entries.length === 0) return;
    showStatus("Exporting CSV...");
    try {
      await shareJournalCsv(entries);
      showStatus("Exported");
    } catch (error) {
      showStatus("Export failed");
    }
  }

  async function onExportPdf() {
    if (entries.length === 0) return;
    showStatus("Generating PDF...");
    try {
      await shareJournalPdf(entries);
      showStatus("Exported");
    } catch (error) {
      showStatus("Export failed");
    }
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <H2>Inner Child Journal</H2>

      {/* Write Section */}
      <Card style={styles.section}>
        <Label>New Entry</Label>
        <Input value={title} onChangeText={setTitle} placeholder="Title (optional)" />
        <Input value={tags} onChangeText={setTags} placeholder="Tags (comma separated)" />
        <Input
          value={content}
          onChangeText={setContent}
          placeholder="Write your thoughts..."
          multiline
          numberOfLines={6}
          style={styles.contentInput}
        />
        <Btn title="Save Entry" onPress={onSave} disabled={!content.trim()} />
        {!!status && <Text style={styles.status}>{status}</Text>}
      </Card>

      {/* Export Section */}
      {canShare && (
        <Card style={styles.section}>
          <Label>Export / Share</Label>
          <View style={styles.exportBtns}>
            <Btn
              title="Share CSV"
              onPress={onExportCsv}
              disabled={entries.length === 0}
              variant="secondary"
            />
            <Btn
              title="Share PDF"
              onPress={onExportPdf}
              disabled={entries.length === 0}
              variant="secondary"
            />
          </View>
          <Text style={styles.hint}>
            Exports are generated on-device and shared via your phone's share sheet.
          </Text>
        </Card>
      )}

      {/* Browse Section */}
      <Card style={styles.section}>
        <Label>Journal Entries ({entries.length})</Label>
        {entries.length === 0 ? (
          <Text style={styles.empty}>No entries yet. Start writing above.</Text>
        ) : (
          entries
            .slice()
            .reverse()
            .map((entry) => (
              <View key={entry.id} style={styles.entry}>
                <Text style={styles.entryTitle}>{entry.title}</Text>
                <Text style={styles.entryMeta}>
                  {new Date(entry.created_at).toLocaleDateString()} &middot;{" "}
                  {entry.tags.join(", ") || "no tags"}
                </Text>
                <Text style={styles.entryContent} numberOfLines={4}>
                  {entry.content}
                </Text>
                <View style={styles.entryActions}>
                  <Btn
                    title="Delete"
                    onPress={() => onDelete(entry.id)}
                    variant="danger"
                    style={styles.smallBtn}
                  />
                </View>
              </View>
            ))
        )}
      </Card>

      {back && (
        <Btn title="Back" onPress={back} variant="secondary" style={styles.backBtn} />
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bgVoid,
  },
  content: {
    padding: 16,
    paddingBottom: 40,
  },
  section: {
    marginBottom: 16,
  },
  contentInput: {
    minHeight: 120,
    textAlignVertical: "top",
  },
  status: {
    marginTop: 12,
    color: colors.accentGreen,
    fontSize: 14,
  },
  exportBtns: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 12,
  },
  hint: {
    fontSize: 12,
    color: colors.textMuted,
  },
  empty: {
    color: colors.textMuted,
    fontStyle: "italic",
  },
  entry: {
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingTop: 12,
    marginTop: 12,
  },
  entryTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: colors.textPrimary,
  },
  entryMeta: {
    fontSize: 12,
    color: colors.textMuted,
    marginTop: 4,
  },
  entryContent: {
    fontSize: 14,
    color: colors.textSecondary,
    marginTop: 8,
    lineHeight: 20,
  },
  entryActions: {
    flexDirection: "row",
    marginTop: 12,
    gap: 8,
  },
  smallBtn: {
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  backBtn: {
    marginTop: 8,
  },
});
