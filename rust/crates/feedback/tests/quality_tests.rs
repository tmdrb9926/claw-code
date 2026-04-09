use feedback::quality::{FeedbackSignal, QualityAnalyzer};

#[test]
fn positive_explicit_approval() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_user_message("좋아, 잘 동작해");
    assert_eq!(signal, FeedbackSignal::Positive);
}

#[test]
fn positive_english_approval() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_user_message("perfect, that works great");
    assert_eq!(signal, FeedbackSignal::Positive);
}

#[test]
fn negative_retry_request() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_user_message("아니, 다시 해줘");
    assert_eq!(signal, FeedbackSignal::Negative);
}

#[test]
fn negative_english_rejection() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_user_message("no, that's wrong, try again");
    assert_eq!(signal, FeedbackSignal::Negative);
}

#[test]
fn neutral_followup_question() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_user_message("이제 테스트 코드도 작성해줘");
    assert_eq!(signal, FeedbackSignal::Neutral);
}

#[test]
fn tool_success_is_low_positive() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_tool_result(/* is_error */ false);
    assert_eq!(signal, FeedbackSignal::WeakPositive);
}

#[test]
fn tool_error_is_weak_negative() {
    let analyzer = QualityAnalyzer::new();
    let signal = analyzer.analyze_tool_result(/* is_error */ true);
    assert_eq!(signal, FeedbackSignal::WeakNegative);
}

#[test]
fn session_overall_quality_positive() {
    let analyzer = QualityAnalyzer::new();
    let signals = vec![
        FeedbackSignal::WeakPositive,
        FeedbackSignal::WeakPositive,
        FeedbackSignal::Positive,
    ];
    assert!(analyzer.is_session_positive(&signals));
}

#[test]
fn session_with_negative_not_positive() {
    let analyzer = QualityAnalyzer::new();
    let signals = vec![
        FeedbackSignal::WeakPositive,
        FeedbackSignal::Negative,
        FeedbackSignal::Neutral,
    ];
    assert!(!analyzer.is_session_positive(&signals));
}
