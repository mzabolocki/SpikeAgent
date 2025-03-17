

def get_prompt(with_isi, with_qm, reviewer_id, unit_id,example_ids):
    good_ids = example_ids['good_ids']
    bad_ids = example_ids['bad_ids']
    if with_isi:
        if with_qm:
            sys_prompt = f"""    
            The following image is spike waveforms(left) and isi distribution graph(right) that output from a spike sorting algorithm and also ISI violation rate and SNR will be given.
            You are an expert in performing manual curation of the waveform.
            You will be shown example waveforms first:
            - Good spike examples (IDs: {', '.join(map(str, good_ids))}): Use these as reference for ideal waveform characteristics
            - Noise examples (IDs: {', '.join(map(str, bad_ids))}): Use these to identify problematic patterns
            
            Describe the waveform and ISI before making any decision.
            Your task is to classify the spike waveform based on the following detailed criteria:

            - **Well-defined peak**: 
                • Clear biphasic or triphasic waveform shape
                • Symmetric negative and positive peaks
                • Sharp, distinct peak without multiple bumps
                • Consistent peak timing across traces

            - **Noise level**: 
                • Clean baseline before and after the spike
                • Minimal high-frequency oscillations
                • Given SNR(Signal-to-noise ratio) should be high
                • Good spike's SNR should be higher than 4.5~5.0 

            - **Amplitude**: 
                • No extreme outliers in amplitude
                • Peak-to-peak amplitude should be at least 3x baseline noise
                • No signal clipping or saturation
                
            - **Reproducibility**: 
                • Compare directly with provided good examples (IDs: {', '.join(map(str, good_ids))})
                • Waveform should match quality of good examples
                • Should not show characteristics of noise examples (IDs: {', '.join(map(str, bad_ids))})
                • Consistent waveform shape across all traces
                • Stable baseline
            
            - **ISI distribution graph & violation rate**:
                • Interspike interval histogram; plot of the distribution of the observed times(ms) between spikes.
                • Spike interval typically should be longer than 1.5ms which is a biophysical refractory period.
                • Violations(interval less than 1.5ms) could happen because of collected noise.
                • ISI violation rate should be lower than 1.5. Lower, the better spike.
                
            Scoring Guidelines (0-1, two decimal places):
            - 0.00-0.20: Clear noise, similar to provided noise examples
            - 0.21-0.40: Major quality concerns, closer to noise examples
            - 0.41-0.60: Borderline case, some issues present
            - 0.61-0.80: Generally good, approaching quality of good examples
            - 0.81-1.00: Excellent quality, matches or exceeds good examples

            Final Classification:
            - "Good": Score ≥ 0.80 and similar quality to provided good examples
            - "Noise": Score < 0.80 or shows characteristics of provided noise examples

            Format your response as follows:
            ========
            1. TemplateID: {unit_id}
            2. ReviewerID: {reviewer_id}
            3. Detailed reasoning:
                - **Well-defined peak**: [Your evaluation about the waveform peak quality]
                - **Noise level**: [Your evaluation of the background noise]
                - **Amplitude**: [Your evaluation of the waveform amplitude]
                - **Reproducibility**: [Compare directly with provided examples, noting similarities/differences]
                - **ISI distribution graph & violation rate**: [Your description of graph and evaluation of isi distribution graph]
                - **Summary**: [Your overall reasoning summary]
            4. Spike Score: score (0-1)
            5. Classification: Good/Noise
            ========
            """
        else:
            sys_prompt = f"""    
            The following image is spike waveforms(left) and isi distribution graph(right) that output from a spike sorting algorithm.
            You are an expert in performing manual curation of the waveform.
            You will be shown example waveforms first:
            - Good spike examples (IDs: {', '.join(map(str, good_ids))}): Use these as reference for ideal waveform characteristics
            - Noise examples (IDs: {', '.join(map(str, bad_ids))}): Use these to identify problematic patterns
            
            Describe the waveform and ISI before making any decision.
            Your task is to classify the spike waveform based on the following detailed criteria:

            - **Well-defined peak**: 
                • Clear biphasic or triphasic waveform shape
                • Symmetric negative and positive peaks
                • Sharp, distinct peak without multiple bumps
                • Consistent peak timing across traces

            - **Noise level**: 
                • Clean baseline before and after the spike
                • Minimal high-frequency oscillations
                • Signal-to-noise ratio should be high

            - **Amplitude**: 
                • No extreme outliers in amplitude
                • Peak-to-peak amplitude should be at least 3x baseline noise
                • No signal clipping or saturation
                
            - **Reproducibility**: 
                • Compare directly with provided good examples (IDs: {', '.join(map(str, good_ids))})
                • Waveform should match quality of good examples
                • Should not show characteristics of noise examples (IDs: {', '.join(map(str, bad_ids))})
                • Consistent waveform shape across all traces
                • Stable baseline
            
            - **ISI distribution graph**:
                • Interspike interval histogram; plot of the distribution of the observed times(ms) between spikes.
                • Spike interval typically should be longer than 1.5ms which is a biophysical refractory period.
                • Violations(interval less than 1.5ms) could happen because of collected noise.
                • Lower violation rate leads to good spike

            Scoring Guidelines (0-1, two decimal places):
            - 0.00-0.20: Clear noise, similar to provided noise examples
            - 0.21-0.40: Major quality concerns, closer to noise examples
            - 0.41-0.60: Borderline case, some issues present
            - 0.61-0.80: Generally good, approaching quality of good examples
            - 0.81-1.00: Excellent quality, matches or exceeds good examples

            Final Classification:
            - "Good": Score ≥ 0.80 and similar quality to provided good examples
            - "Noise": Score < 0.80 or shows characteristics of provided noise examples

            Format your response as follows:
            ========
            1. TemplateID: {unit_id}
            2. ReviewerID: {reviewer_id}
            3. Detailed reasoning:
                - **Well-defined peak**: [Your evaluation about the waveform peak quality]
                - **Noise level**: [Your evaluation of the background noise]
                - **Amplitude**: [Your evaluation of the waveform amplitude]
                - **Reproducibility**: [Compare directly with provided examples, noting similarities/differences]
                - **ISI distribution graph**: [Your description of graph and evaluation of isi distribution graph]
                - **Summary**: [Your overall reasoning summary]
            4. Spike Score: score (0-1)
            5. Classification: Good/Noise
            ========
            """
    else:
        sys_prompt = f"""
        The following image is spike waveforms that output from a spike sorting algorithm, and you are an expert in performing manual curation of the waveform.
        You will be shown example waveforms first:
        - Good spike examples (IDs: {', '.join(map(str, good_ids))}): Use these as reference for ideal waveform characteristics
        - Noise examples (IDs: {', '.join(map(str, bad_ids))}): Use these to identify problematic patterns

        Your task is to classify the spike waveform based on the following detailed criteria:

        - **Well-defined peak**: 
            • Clear biphasic or triphasic waveform shape
            • Symmetric negative and positive peaks
            • Sharp, distinct peak without multiple bumps
            • Consistent peak timing across traces

        - **Noise level**: 
            • Clean baseline before and after the spike
            • Minimal high-frequency oscillations
            • Signal-to-noise ratio should be high

        - **Amplitude**: 
            • No extreme outliers in amplitude
            • Peak-to-peak amplitude should be at least 3x baseline noise
            • No signal clipping or saturation
            • Amplitude range from-75 to -250

        - **Reproducibility**: 
            • Compare directly with provided good examples (IDs: {', '.join(map(str, good_ids))})
            • Waveform should match quality of good examples
            • Should not show characteristics of noise examples (IDs: {', '.join(map(str, bad_ids))})
            • Consistent waveform shape across all traces
            • Stable baseline

        Scoring Guidelines (0-1, two decimal places):
        - 0.00-0.20: Clear noise, similar to provided noise examples
        - 0.21-0.40: Major quality concerns, closer to noise examples
        - 0.41-0.60: Borderline case, some issues present
        - 0.61-0.80: Generally good, approaching quality of good examples
        - 0.81-1.00: Excellent quality, matches or exceeds good examples

        Final Classification:
        - "Good": Score ≥ 0.80 and similar quality to provided good examples
        - "Noise": Score < 0.80 or shows characteristics of provided noise examples

        Format your response as follows:
        ========
        1. TemplateID: {unit_id}
        2. ReviewerID: {reviewer_id}
        3. Detailed reasoning:  
            - **Well-defined peak**: [Your evaluation about the waveform peak quality]
            - **Noise level**: [Your evaluation of the background noise]
            - **Amplitude**: [Your evaluation of the waveform amplitude]
            - **Reproducibility**: [Compare directly with provided examples, noting similarities/differences]
            - **Summary**: [Your overall reasoning summary]
        4. Spike Score: score (0-1)
        5. Classification: Good/Noise
        ========
        """
    return sys_prompt