/**
 * LUMINARK Mobile UI Components
 * Reusable UI primitives for the Inner Child Journal app
 */

import React from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  TextInputProps,
  ViewStyle,
  TextStyle,
} from "react-native";

// LUMINARK color palette
export const colors = {
  bgVoid: "#0a0a0a",
  bgPrimary: "#111111",
  bgSecondary: "#1a1a1a",
  accentGreen: "#00ff41",
  accentCyan: "#00d4ff",
  textPrimary: "#ffffff",
  textSecondary: "#cccccc",
  textMuted: "#888888",
  border: "#333333",
  error: "#ff3366",
  warning: "#ffaa00",
};

interface BtnProps {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary" | "danger";
  style?: ViewStyle;
}

export function Btn({ title, onPress, disabled, variant = "primary", style }: BtnProps) {
  const bgColor =
    variant === "primary"
      ? colors.accentGreen
      : variant === "danger"
      ? colors.error
      : colors.bgSecondary;

  const textColor = variant === "primary" ? colors.bgVoid : colors.textPrimary;

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled}
      style={[
        styles.btn,
        { backgroundColor: disabled ? colors.border : bgColor },
        style,
      ]}
    >
      <Text style={[styles.btnText, { color: disabled ? colors.textMuted : textColor }]}>
        {title}
      </Text>
    </TouchableOpacity>
  );
}

interface CardProps {
  children: React.ReactNode;
  style?: ViewStyle;
}

export function Card({ children, style }: CardProps) {
  return <View style={[styles.card, style]}>{children}</View>;
}

interface H2Props {
  children: React.ReactNode;
  style?: TextStyle;
}

export function H2({ children, style }: H2Props) {
  return <Text style={[styles.h2, style]}>{children}</Text>;
}

interface InputProps extends TextInputProps {
  style?: ViewStyle;
}

export function Input(props: InputProps) {
  return (
    <TextInput
      placeholderTextColor={colors.textMuted}
      {...props}
      style={[styles.input, props.style]}
    />
  );
}

interface LabelProps {
  children: React.ReactNode;
  style?: TextStyle;
}

export function Label({ children, style }: LabelProps) {
  return <Text style={[styles.label, style]}>{children}</Text>;
}

const styles = StyleSheet.create({
  btn: {
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
    minWidth: 100,
  },
  btnText: {
    fontSize: 16,
    fontWeight: "600",
  },
  card: {
    backgroundColor: colors.bgSecondary,
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: colors.border,
  },
  h2: {
    fontSize: 24,
    fontWeight: "700",
    color: colors.textPrimary,
    marginBottom: 16,
  },
  input: {
    backgroundColor: colors.bgPrimary,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: colors.textPrimary,
    marginBottom: 12,
  },
  label: {
    fontSize: 14,
    fontWeight: "600",
    color: colors.textSecondary,
    marginBottom: 6,
  },
});
