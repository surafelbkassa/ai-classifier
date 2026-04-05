package classifier

import "strings"

func Classify(text string) (string, string) {
	text = strings.ToLower(text)

	if strings.Contains(text, "urgent") || strings.Contains(text, "meeting") || strings.Contains(text, "deadline") {
		return "work", "high"
	}

	if strings.Contains(text, "free") || strings.Contains(text, "offer") || strings.Contains(text, "sale") {
		return "spam", "low"
	}
	return "personal", "medium"

}
