use serde::{Deserialize, Serialize};

/// Strength and direction of a feedback signal extracted from a single event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackSignal {
    /// Explicit positive language ("좋아", "perfect", "thanks").
    Positive,
    /// Tool executed successfully without error.
    WeakPositive,
    /// No clear signal -- follow-up question or neutral statement.
    Neutral,
    /// Tool execution failed.
    WeakNegative,
    /// Explicit negative language ("아니", "wrong", "다시").
    Negative,
}

impl FeedbackSignal {
    /// Numeric weight used when aggregating signals for a full session.
    #[must_use]
    pub fn weight(self) -> i32 {
        match self {
            Self::Positive => 3,
            Self::WeakPositive => 1,
            Self::Neutral => 0,
            Self::WeakNegative => -1,
            Self::Negative => -3,
        }
    }
}

/// Analyzes conversation events for implicit and explicit feedback signals.
#[derive(Debug, Clone)]
pub struct QualityAnalyzer {
    positive_patterns: Vec<&'static str>,
    negative_patterns: Vec<&'static str>,
}

impl QualityAnalyzer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            positive_patterns: vec![
                // Korean
                "좋아",
                "완벽",
                "잘 동작",
                "고마워",
                "감사",
                "맞아",
                "좋네",
                "훌륭",
                // English
                "perfect",
                "great",
                "thanks",
                "works",
                "good",
                "excellent",
                "nice",
                "awesome",
                "correct",
                "exactly",
            ],
            negative_patterns: vec![
                // Korean
                "아니",
                "다시",
                "틀렸",
                "잘못",
                "아닌데",
                "안 돼",
                "에러",
                "고쳐",
                // English
                "wrong",
                "no,",
                "try again",
                "incorrect",
                "fix",
                "broken",
                "doesn't work",
                "not right",
                "redo",
            ],
        }
    }

    /// Classify a user message as positive, negative, or neutral.
    #[must_use]
    pub fn analyze_user_message(&self, text: &str) -> FeedbackSignal {
        let lower = text.to_lowercase();

        // Check negative first -- explicit rejection is a strong signal.
        for pattern in &self.negative_patterns {
            if lower.contains(pattern) {
                return FeedbackSignal::Negative;
            }
        }

        for pattern in &self.positive_patterns {
            if lower.contains(pattern) {
                return FeedbackSignal::Positive;
            }
        }

        FeedbackSignal::Neutral
    }

    /// Classify a tool execution result.
    #[must_use]
    pub fn analyze_tool_result(&self, is_error: bool) -> FeedbackSignal {
        if is_error {
            FeedbackSignal::WeakNegative
        } else {
            FeedbackSignal::WeakPositive
        }
    }

    /// Determine if a session's aggregated signals indicate positive quality.
    ///
    /// A session is considered positive if:
    /// 1. No explicit `Negative` signal exists, AND
    /// 2. The weighted sum of all signals is strictly positive.
    #[must_use]
    pub fn is_session_positive(&self, signals: &[FeedbackSignal]) -> bool {
        let has_negative = signals.contains(&FeedbackSignal::Negative);
        if has_negative {
            return false;
        }
        let total_weight: i32 = signals.iter().map(|s| s.weight()).sum();
        total_weight > 0
    }
}

impl Default for QualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
