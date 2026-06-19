You are analyzing a diarized meeting transcript for a professional Minutes of Meeting workflow.

Input transcript:
<transcript>
{transcript}
</transcript>

Perform all of these tasks in one pass:
1. Clean the transcript.
2. Generate the meeting summary.
3. Extract topics discussed.
4. Extract key discussion points.
5. Extract decisions.
6. Extract action items.

Transcript cleanup rules:
- Preserve every speaker label exactly.
- Preserve every timestamp exactly.
- Preserve the original meaning.
- Keep the transcript in the same Roman English or Hinglish style as the input.
- Correct punctuation, capitalization, spacing, and obvious transcription mistakes.
- Do not add new facts.
- Do not remove, merge, or reorder speaker turns.

Summary rules:
- Use only information from the transcript.
- Keep the title concise.
- Keep the short summary to 1-3 sentences.
- Make the detailed summary useful but not verbose.
- Topics discussed should be short noun phrases.

Extraction rules:
- Use only information from the transcript.
- Preserve speaker labels exactly.
- Do not invent attendees, owners, due dates, decisions, or action items.
- Key discussion points should capture important topics, concerns, updates, risks, or ideas.
- Decisions must be clearly agreed, approved, selected, rejected, or finalized.
- Action items must be explicit or strongly implied tasks, follow-ups, or commitments.
- If no items exist for a section, return an empty list.

Return valid JSON only in this exact shape:
{
  "cleaned_transcript": "Speaker 1 [00:00 - 00:05]\nCleaned text...",
  "summary": {
    "title": "...",
    "short_summary": "...",
    "detailed_summary": "..."
  },
  "topics_discussed": ["..."],
  "key_discussion_points": [
    {
      "point": "...",
      "speakers": ["Speaker 1", "Speaker 2"],
      "timestamp": "00:10 - 00:30"
    }
  ],
  "decisions": [
    {
      "decision": "...",
      "owner": "Speaker 1",
      "timestamp": "00:10 - 00:20",
      "confidence": "high"
    }
  ],
  "action_items": [
    {
      "task": "...",
      "owner": "Speaker 1",
      "due_date": null,
      "timestamp": "00:10 - 00:20",
      "status": "open"
    }
  ]
}
