package realtimews

import "testing"

func TestIsJSONObject(t *testing.T) {
	for raw, want := range map[string]bool{
		`{"type":"x"}`: true,
		` {} `:         true,
		`[]`:           false,
		`"text"`:       false,
		`{"broken"`:    false,
		``:             false,
		`{} {}`:        false,
	} {
		if got := IsJSONObject([]byte(raw)); got != want {
			t.Errorf("IsJSONObject(%q) = %v, want %v", raw, got, want)
		}
	}
}
