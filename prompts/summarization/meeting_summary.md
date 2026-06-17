You are generating a meeting summary from a cleaned diarized transcript.

Input transcript:
<transcript>
{transcript}
</transcript>

Generate:
- Meeting title
- Short summary
- Detailed summary
- Topics discussed

Rules:
- Use only information from the transcript.
- Do not invent attendees, dates, decisions, or action items.
- Keep the title concise.
- Keep the short summary to 1-3 sentences.
- Make the detailed summary useful but not verbose.
- Topics discussed should be short noun phrases.

Return valid JSON only in this exact shape:
{
  "title": "...",
  "short_summary": "...",
  "detailed_summary": "...",
  "topics_discussed": ["..."]
}
