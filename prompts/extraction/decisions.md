You are extracting decisions from a cleaned diarized meeting transcript.

Input transcript:
<transcript>
{transcript}
</transcript>

Extract only decisions that were clearly agreed, approved, selected, rejected, or finalized.

For each decision, return:
- decision: the concrete decision made
- owner: the person or speaker responsible for the decision area, or null if unknown
- timestamp: the timestamp range where the decision appears, or null if unknown
- confidence: "high", "medium", or "low" based on how explicit the transcript is

Rules:
- Use only information from the transcript.
- Do not invent decisions.
- Do not extract general discussion points.
- Do not extract action items unless they are also explicit decisions.
- Preserve speaker labels exactly when using them as owners.
- If no decisions exist, return an empty list.

Return valid JSON only in this exact shape:
{
  "decisions": [
    {
      "decision": "...",
      "owner": "Speaker 1",
      "timestamp": "00:10 - 00:20",
      "confidence": "high"
    }
  ]
}
