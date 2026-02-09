"""
Alignment module to fuse ASR transcription segments with speaker diarization segments.
Assigns speaker labels to text segments based on timestamp overlap.
"""

from typing import List, Dict
import numpy as np


def calculate_overlap(seg1_start: float, seg1_end: float, 
                     seg2_start: float, seg2_end: float) -> float:
    """
    Calculate overlap ratio between two time segments.
    
    Args:
        seg1_start, seg1_end: First segment boundaries
        seg2_start, seg2_end: Second segment boundaries
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    overlap_start = max(seg1_start, seg2_start)
    overlap_end = min(seg1_end, seg2_end)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_duration = overlap_end - overlap_start
    seg1_duration = seg1_end - seg1_start
    
    if seg1_duration == 0:
        return 0.0
    
    return overlap_duration / seg1_duration


def align_segments(asr_segments: List[Dict], diarization_segments: List[Dict],
                   overlap_threshold: float = 0.5) -> List[Dict]:
    """
    Align ASR segments with diarization segments based on timestamp overlap.
    
    Args:
        asr_segments: List of ASR segments with 'start', 'end', 'text'
        diarization_segments: List of diarization segments with 'start', 'end', 'speaker_label'
        overlap_threshold: Minimum overlap ratio to assign speaker (default: 0.5)
        
    Returns:
        List of aligned segments with 'start', 'end', 'text', 'speaker_label'
    """
    aligned_segments = []
    
    for asr_seg in asr_segments:
        asr_start = asr_seg['start']
        asr_end = asr_seg['end']
        asr_text = asr_seg['text']
        
        # Find best matching diarization segment
        best_speaker = None
        best_overlap = 0.0
        
        for diar_seg in diarization_segments:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            diar_speaker = diar_seg['speaker_label']
            
            # Calculate overlap
            overlap = calculate_overlap(asr_start, asr_end, diar_start, diar_end)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_speaker
        
        # Assign speaker if overlap exceeds threshold
        if best_overlap >= overlap_threshold:
            speaker_label = best_speaker
        else:
            # If no good match, try to find any overlapping segment
            speaker_label = None
            for diar_seg in diarization_segments:
                diar_start = diar_seg['start']
                diar_end = diar_seg['end']
                if not (asr_end <= diar_start or asr_start >= diar_end):
                    speaker_label = diar_seg['speaker_label']
                    break
            
            if speaker_label is None:
                speaker_label = "UNKNOWN"
        
        aligned_segments.append({
            'start': asr_start,
            'end': asr_end,
            'text': asr_text,
            'speaker_label': speaker_label
        })
    
    return aligned_segments


def merge_adjacent_segments(aligned_segments: List[Dict], 
                           same_speaker_threshold: float = 0.5) -> List[Dict]:
    """
    Merge adjacent segments from the same speaker if gap is small.
    
    Args:
        aligned_segments: List of aligned segments
        same_speaker_threshold: Maximum gap in seconds to merge (default: 0.5)
        
    Returns:
        List of merged segments
    """
    if not aligned_segments:
        return []
    
    merged = []
    current_seg = aligned_segments[0].copy()
    
    for next_seg in aligned_segments[1:]:
        # Check if same speaker and gap is small
        if (current_seg['speaker_label'] == next_seg['speaker_label'] and
            next_seg['start'] - current_seg['end'] <= same_speaker_threshold):
            # Merge segments
            current_seg['end'] = next_seg['end']
            current_seg['text'] += " " + next_seg['text']
        else:
            # Save current segment and start new one
            merged.append(current_seg)
            current_seg = next_seg.copy()
    
    # Add last segment
    merged.append(current_seg)
    
    return merged


def format_aligned_output(segments: List[Dict]) -> str:
    """
    Format aligned segments as human-readable string.
    
    Args:
        segments: List of aligned segments
        
    Returns:
        Formatted string
    """
    output = []
    for seg in segments:
        output.append(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker_label']}: {seg['text']}")
    return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    asr_segments = [
        {'start': 0.0, 'end': 2.5, 'text': 'Hello, how are you?'},
        {'start': 2.5, 'end': 5.0, 'text': 'I am doing well, thank you.'}
    ]
    
    diarization_segments = [
        {'start': 0.0, 'end': 2.8, 'speaker_label': 'SPEAKER_00'},
        {'start': 2.8, 'end': 5.2, 'speaker_label': 'SPEAKER_01'}
    ]
    
    aligned = align_segments(asr_segments, diarization_segments)
    print("Aligned segments:")
    print(format_aligned_output(aligned))
