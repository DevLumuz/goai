package google

import (
	"strconv"
	"time"
)

// LiveConfig holds the Gemini Live settings without a common realtime field,
// as in the official SDK's LiveConnectConfig; zero values keep the defaults.
// Use it with goai.WithRealtimeProviderOptions(google.LiveProviderOptions(cfg)).
type LiveConfig struct {
	// ResponseModalities overrides the default AUDIO, for models that also
	// answer in text.
	ResponseModalities []string
	Temperature        *float64
	TopP               *float64
	TopK               *float64
	Seed               *int
	// MediaResolution is MEDIA_RESOLUTION_LOW, _MEDIUM, or _HIGH for image and
	// video input.
	MediaResolution string

	// LanguageCode is the BCP-47 language of the generated speech.
	LanguageCode string
	// ReplicatedVoice speaks with a cloned voice, verified by a consent recording.
	ReplicatedVoice *LiveReplicatedVoice
	Translation     *LiveTranslation

	// Thinking and AffectiveDialog apply to models that support them, such as
	// Gemini 2.5; Gemini 3.8 Live rejects both.
	Thinking        *LiveThinking
	AffectiveDialog *bool
	// ProactiveAudio lets the model stay silent when the input is not for it.
	ProactiveAudio *bool

	SessionResumption  *LiveSessionResumption
	ContextCompression *LiveContextCompression
	// InitialHistoryInClientContent makes the server take conversation history
	// (RealtimeSession.AppendConversation) before the realtime input starts.
	InitialHistoryInClientContent bool
	// ExplicitVADSignal asks the server for voiceActivityDetectionSignal
	// messages. Vertex AI only.
	ExplicitVADSignal bool
	Avatar            *LiveAvatar
	SafetySettings    []LiveSafetySetting
	// Labels attach user metadata to the session; Vertex AI also uses them
	// for billing and reporting.
	Labels map[string]string

	ActivityDetection *LiveActivityDetection
	// TurnCoverage selects which input belongs to the user's turn, such as
	// TURN_INCLUDES_ONLY_ACTIVITY or TURN_INCLUDES_ALL_INPUT.
	TurnCoverage string
	// InterimTranscriptTimestamps adds timestamps to interim transcripts.
	InterimTranscriptTimestamps *bool
	// InputTranscription and OutputTranscription tune transcription; setting
	// one enables transcription for that side.
	InputTranscription  *LiveTranscription
	OutputTranscription *LiveTranscription
}

// LiveReplicatedVoice clones a voice from a sample. The first session sends the
// consent recording; later ones can reuse the signature from
// RealtimeSessionReady.ProviderMetadata["google"]["voiceConsentSignature"].
// ConsentAudio and ConsentSignature are Gemini API only.
type LiveReplicatedVoice struct {
	MIMEType         string
	Sample           []byte
	ConsentAudio     []byte
	ConsentSignature string
}

// LiveTranslation speaks the reply in TargetLanguage.
type LiveTranslation struct {
	TargetLanguage     string
	EchoTargetLanguage *bool
}

// LiveThinking configures model reasoning; Level is LOW, MEDIUM, or HIGH.
type LiveThinking struct {
	Budget          *int
	IncludeThoughts bool
	Level           string
}

// LiveSessionResumption resumes the session of Handle, or starts a new one when
// empty; the server then reports resume tokens. Transparent also reports the
// last client message each token covers, so a reconnect can resend the rest.
// Transparent is Vertex AI only.
type LiveSessionResumption struct {
	Handle      string
	Transparent bool
}

// LiveContextCompression keeps long sessions within the context window with a
// sliding window: past TriggerTokens, history shrinks to TargetTokens. Zero
// values use the server defaults.
type LiveContextCompression struct {
	TriggerTokens int64
	TargetTokens  int64
}

// LiveAvatar renders the model as a video avatar, either prebuilt by Name or
// from a 9:16 reference image. Bitrates are in bits per second.
type LiveAvatar struct {
	Name          string
	ImageMIMEType string
	Image         []byte
	AudioBitrate  int
	VideoBitrate  int
}

// LiveSafetySetting blocks unsafe content of Category at Threshold. Method
// (SEVERITY or PROBABILITY) is Vertex AI only.
type LiveSafetySetting struct {
	Category  string
	Threshold string
	Method    string
}

// LiveActivityDetection tunes the server voice activity detection.
// Sensitivities are START_SENSITIVITY_HIGH/LOW and END_SENSITIVITY_HIGH/LOW.
type LiveActivityDetection struct {
	StartSensitivity string
	EndSensitivity   string
	PrefixPadding    time.Duration
	SilenceDuration  time.Duration
}

// LiveTranscription tunes one side of the transcription. Mode is VERBATIM or
// SMART.
type LiveTranscription struct {
	LanguageCodes     []string
	LanguageAuto      bool
	LanguageHints     []string
	CustomVocabulary  []string
	AdaptationPhrases []string
	Diarization       *bool
	WordTimestamps    *bool
	Mode              string
}

// LiveProviderOptions converts cfg to the ProviderOptions of a Gemini Live
// session, in Live setup field names.
func LiveProviderOptions(cfg LiveConfig) map[string]any {
	setup := map[string]any{}
	gen := map[string]any{}
	putSlice(gen, "responseModalities", cfg.ResponseModalities)
	putPtr(gen, "temperature", cfg.Temperature)
	putPtr(gen, "topP", cfg.TopP)
	putPtr(gen, "topK", cfg.TopK)
	putPtr(gen, "seed", cfg.Seed)
	putString(gen, "mediaResolution", cfg.MediaResolution)
	putPtr(gen, "enableAffectiveDialog", cfg.AffectiveDialog)
	if speech := cfg.speechConfig(); len(speech) > 0 {
		gen["speechConfig"] = speech
	}
	if t := cfg.Thinking; t != nil {
		thinking := map[string]any{}
		putPtr(thinking, "thinkingBudget", t.Budget)
		putBool(thinking, "includeThoughts", t.IncludeThoughts)
		putString(thinking, "thinkingLevel", t.Level)
		gen["thinkingConfig"] = thinking
	}
	if t := cfg.Translation; t != nil {
		translation := map[string]any{}
		putString(translation, "targetLanguageCode", t.TargetLanguage)
		putPtr(translation, "echoTargetLanguage", t.EchoTargetLanguage)
		gen["translationConfig"] = translation
	}
	if len(gen) > 0 {
		setup["generationConfig"] = gen
	}

	if cfg.ProactiveAudio != nil {
		setup["proactivity"] = map[string]any{"proactiveAudio": *cfg.ProactiveAudio}
	}
	if r := cfg.SessionResumption; r != nil {
		resumption := map[string]any{}
		putString(resumption, "handle", r.Handle)
		putBool(resumption, "transparent", r.Transparent)
		setup["sessionResumption"] = resumption
	}
	if c := cfg.ContextCompression; c != nil {
		// int64 fields travel as JSON strings, as in the official SDK.
		window := map[string]any{}
		if c.TargetTokens > 0 {
			window["targetTokens"] = strconv.FormatInt(c.TargetTokens, 10)
		}
		compression := map[string]any{"slidingWindow": window}
		if c.TriggerTokens > 0 {
			compression["triggerTokens"] = strconv.FormatInt(c.TriggerTokens, 10)
		}
		setup["contextWindowCompression"] = compression
	}
	if cfg.InitialHistoryInClientContent {
		setup["historyConfig"] = map[string]any{"initialHistoryInClientContent": true}
	}
	putBool(setup, "explicitVadSignal", cfg.ExplicitVADSignal)
	if a := cfg.Avatar; a != nil {
		avatar := map[string]any{}
		putString(avatar, "avatarName", a.Name)
		if len(a.Image) > 0 {
			avatar["customizedAvatar"] = map[string]any{"imageMimeType": a.ImageMIMEType, "imageData": a.Image}
		}
		putInt(avatar, "audioBitrateBps", a.AudioBitrate)
		putInt(avatar, "videoBitrateBps", a.VideoBitrate)
		setup["avatarConfig"] = avatar
	}
	if len(cfg.SafetySettings) > 0 {
		settings := make([]map[string]any, 0, len(cfg.SafetySettings))
		for _, s := range cfg.SafetySettings {
			setting := map[string]any{}
			putString(setting, "category", s.Category)
			putString(setting, "threshold", s.Threshold)
			putString(setting, "method", s.Method)
			settings = append(settings, setting)
		}
		setup["safetySettings"] = settings
	}
	if len(cfg.Labels) > 0 {
		labels := make(map[string]any, len(cfg.Labels))
		for k, v := range cfg.Labels {
			labels[k] = v
		}
		setup["labels"] = labels
	}

	input := map[string]any{}
	if d := cfg.ActivityDetection; d != nil {
		detection := map[string]any{}
		putString(detection, "startOfSpeechSensitivity", d.StartSensitivity)
		putString(detection, "endOfSpeechSensitivity", d.EndSensitivity)
		putInt(detection, "prefixPaddingMs", int(d.PrefixPadding.Milliseconds()))
		putInt(detection, "silenceDurationMs", int(d.SilenceDuration.Milliseconds()))
		input["automaticActivityDetection"] = detection
	}
	putString(input, "turnCoverage", cfg.TurnCoverage)
	putPtr(input, "interimTranscriptTimestampEnabled", cfg.InterimTranscriptTimestamps)
	if len(input) > 0 {
		setup["realtimeInputConfig"] = input
	}
	if t := cfg.InputTranscription; t != nil {
		setup["inputAudioTranscription"] = t.wire()
	}
	if t := cfg.OutputTranscription; t != nil {
		setup["outputAudioTranscription"] = t.wire()
	}
	return map[string]any{"google": setup}
}

func (cfg LiveConfig) speechConfig() map[string]any {
	speech := map[string]any{}
	putString(speech, "languageCode", cfg.LanguageCode)
	if r := cfg.ReplicatedVoice; r != nil {
		replicated := map[string]any{}
		putString(replicated, "mimeType", r.MIMEType)
		if len(r.Sample) > 0 {
			replicated["voiceSampleAudio"] = r.Sample
		}
		if len(r.ConsentAudio) > 0 {
			replicated["consentAudio"] = r.ConsentAudio
		}
		if r.ConsentSignature != "" {
			replicated["voiceConsentSignature"] = map[string]any{"signature": r.ConsentSignature}
		}
		speech["voiceConfig"] = map[string]any{"replicatedVoiceConfig": replicated}
	}
	return speech
}

func (t LiveTranscription) wire() map[string]any {
	tc := map[string]any{}
	putSlice(tc, "languageCodes", t.LanguageCodes)
	if t.LanguageAuto {
		tc["languageAuto"] = map[string]any{}
	}
	if len(t.LanguageHints) > 0 {
		tc["languageHints"] = map[string]any{"languageCodes": t.LanguageHints}
	}
	putSlice(tc, "customVocabulary", t.CustomVocabulary)
	putSlice(tc, "adaptationPhrases", t.AdaptationPhrases)
	putPtr(tc, "diarization", t.Diarization)
	putPtr(tc, "wordTimestamp", t.WordTimestamps)
	putString(tc, "mode", t.Mode)
	return tc
}

func putString(m map[string]any, key, value string) {
	if value != "" {
		m[key] = value
	}
}

func putBool(m map[string]any, key string, value bool) {
	if value {
		m[key] = true
	}
}

func putInt(m map[string]any, key string, value int) {
	if value > 0 {
		m[key] = value
	}
}

func putPtr[T any](m map[string]any, key string, value *T) {
	if value != nil {
		m[key] = *value
	}
}

func putSlice(m map[string]any, key string, value []string) {
	if len(value) > 0 {
		m[key] = append([]string(nil), value...)
	}
}
