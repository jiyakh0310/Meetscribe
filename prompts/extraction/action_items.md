You are extracting action items from a cleaned diarized meeting transcript.

Input transcript:
<transcript>
{transcript}
</transcript>

Extract only explicit or strongly implied tasks, follow-ups, or commitments.

For each action item, return:
- task: the concrete work to be done
- owner: the responsible person or speaker label, or null if unknown
- due_date: the due date or timeline mentioned, or null if unknown
- timestamp: the timestamp range where the action item appears, or null if unknown
- status: usually "open" unless the transcript clearly says it is completed

Rules:
- Use only information from the transcript.
- Do not invent owners or due dates.
- Preserve speaker labels exactly when using them as owners.
- Keep tasks concise and actionable.
- If no action items exist, return an empty list.

Return valid JSON only in this exact shape:
{
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
