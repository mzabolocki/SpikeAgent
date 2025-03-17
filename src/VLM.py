import asyncio
import base64
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from custom_class import ChatAnthropic_H
from pydantic import BaseModel, Field
from typing import Literal
from statistics import mean
from collections import Counter
import nest_asyncio
from tqdm.asyncio import tqdm
from pathlib import Path
import sys
from VLM_prompt import get_prompt
# Import and load environment variables
from dotenv import load_dotenv
load_dotenv()

nest_asyncio.apply()

MODEL_ID = {
    "claude_3_5_sonnet": ChatAnthropic_H(model_name="claude-3-5-sonnet-20240620-v1",temperature=0.7),
    "claude_3_opus": ChatAnthropic_H(model_name="claude-3-opus-20240229-v1",temperature=0.7),
    "claude_3_haiku": ChatAnthropic_H(model_name="claude-3-haiku-20240307-v1",temperature=0.7),
    "claude_3_sonnet": ChatAnthropic_H(model_name="claude-3-sonnet-20240229-v1",temperature=0.7),
    "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0.7),
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
    "o1": ChatOpenAI(model="o1", temperature=0.7),
    "gpt-4-turbo": ChatOpenAI(model="gpt-4-turbo", temperature=0.7),
    "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
    "gemini_2_0_flash": ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7),
    "gemini_1_5_flash": ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7),
    "gemini_1_5_flash_8b": ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.7),
    "gemini_1_5_pro": ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
}

N_SEMAPHORE = {"gpt-4o":20,
        "gpt-4-turbo":20,
        "claude_3_5_sonnet":5,
        "claude_3_sonnet":5,
        "gpt-4o-mini":10,
        "o1":10,
        "gpt-3.5-turbo":10,
        "claude_3_opus":1,
        "claude_3_haiku":5,
        "gemini_2_0_flash":1, 
        "gemini_1_5_flash":20, 
        "gemini_1_5_flash_8b":20, 
        "gemini_1_5_pro":20}

PRE_REJECTED_DEFAULT = {"average_score":[0.0], 
             "final_classification":"Bad", 
             "combined_reasoning":"Pre_rejected",
                "reviewer_1_score": [0.0], 
                "reviewer_2_score":[0.0], 
                "reviewer_3_score":[0.0],
                "reviewer_1_class":"Noise", 
                "reviewer_2_class":"Noise", 
                "reviewer_3_class":"Noise"}


def get_all_unit_ids(waveform_plot_folder):
    """
    Retrieve all unit IDs by scanning the waveform_plot_folder for images named 'extremum_templates_#.png'.

    Args:
        waveform_plot_folder (str): Path to the folder containing template images.

    Returns:
        list: List of unit IDs (integers) derived from filenames.
    """
    unit_ids = []
    for file in os.listdir(waveform_plot_folder):
        if file.startswith("extremum_templates_") and file.endswith(".png"):
            try:
                # Extract the unit ID by removing the prefix and the .png extension
                unit_id_str = file.replace("extremum_templates_", "").replace(".png", "")
                unit_id = int(unit_id_str)
                unit_ids.append(unit_id)
            except ValueError:
                print(f"Skipping invalid file: {file}")
    return sorted(unit_ids)

# Define the structured output for spike waveform classification
class SpikeClassification(BaseModel):
    """Structured output for spike waveform classification."""

    template_id: int = Field(description="The ID of the waveform template being classified.")
    reviewer_id: int = Field(description="The ID of the reviewer making the classification.")
    reasoning: str = Field(description="Detailed reasoning from multiple perspectives for the classification. Please be specific!")
    spike_score: float = Field(description="Score of the spike from 0 to 1 (two decimal places).")
    classification: Literal["Good", "Noise"] = Field(
        description="The classification of the spike as Good, Noise."
    )
    

# Asynchronous function to encode the image of a given unit_id to a base64 string
def get_base64_image(unit_id, waveform_plot_folder):
    image_path = f"{waveform_plot_folder}/extremum_templates_{unit_id}.png"
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image for unit_id {unit_id}: {e}")
        return None
    
def get_content(example_ids, unit_id, waveform_plot_folder, with_qm):
    good_ids = example_ids['good_ids']
    bad_ids = example_ids['bad_ids']
    
    content1 = [{"type": "text", "text": "These are good spikes"}]
    for ids in good_ids:
        content1 += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(ids, waveform_plot_folder)}"}}]
        
    content2 = [{"type": "text", "text": "These are noise"}]
    for ids in bad_ids:
        content2 += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(ids, waveform_plot_folder)}"}}]

    content3=[
            {"type": "text", "text": "These are the spikes for you to determine:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(unit_id, waveform_plot_folder)}"}},
        ]
    
    if with_qm:
        mp = waveform_plot_folder.parent / 'quality_metrics' / 'metrics.csv'
        metrics = pd.read_csv(mp)
        snr = metrics[metrics.iloc[:,0] == unit_id]['snr'].values[0]
        isi = metrics[metrics.iloc[:,0] == unit_id]['isi_violations_ratio'].values[0]
        content4 = [{"type": "text", "text": "These are ISI viloation rate and SNR(signal to noise ratio) of this unit"},
                    {"type": "text", "text": f"ISI viloation rate:{isi}%, SNR: {snr}"}]
        return content1, content2, content3, content4
    
    
    return content1, content2, content3


# Asynchronous function to classify a spike waveform independently
async def classify_waveform_independently(unit_id, reviewer_id, waveform_plot_folder, model, example_ids, with_isi, with_qm):
    await asyncio.sleep(1)
    structured_llm = model.with_structured_output(SpikeClassification)
    
    sys_prompt = get_prompt(with_isi, with_qm, reviewer_id, unit_id, example_ids)
    sys_message = SystemMessage(content=sys_prompt)
    contents = get_content(example_ids, unit_id, waveform_plot_folder, with_qm)

    # Example messages (good and noise spikes for context)
    message_1 = HumanMessage(content=contents[0],)

    message_2 = HumanMessage(content=contents[1],)

    # Main message with the unit_id waveform
    message_3 = HumanMessage(content=contents[2])
    
    if with_qm:
        message_4 = HumanMessage(content=contents[3])
        response = await structured_llm.ainvoke([sys_message, message_1, message_2, message_3, message_4])
    else:
        response = await structured_llm.ainvoke([sys_message, message_1, message_2, message_3])
    return {"reviewer_id": reviewer_id, "response": response}


# Function to aggregate results from all reviewers
def aggregate_results(reviews, unit_id):
    spike_scores = [r["response"].spike_score for r in reviews]
    classifications = [r["response"].classification for r in reviews]

    # Average spike score
    average_score = round(mean(spike_scores), 2)

    # Majority vote for classification
    good_count = Counter(classifications)['Good']
    if good_count == 3:
        majority_classification = 'Good'
    elif good_count == 2 or good_count == 1:
        majority_classification = 'Fair'
    else:
        majority_classification = 'Bad'

    # Combine reasoning for transparency
    combined_reasoning = "\n".join([f"Reviewer {r['reviewer_id']}: {r['response'].reasoning}" for r in reviews])

    # Extract individual results
    individual_scores = {f"reviewer_{r['reviewer_id']}_score": r["response"].spike_score for r in reviews}
    individual_classes = {f"reviewer_{r['reviewer_id']}_class": r["response"].classification for r in reviews}

    return {
        "template_id": unit_id,
        "average_score": average_score,
        "final_classification": majority_classification,
        "combined_reasoning": combined_reasoning,
        **individual_scores,
        **individual_classes
    }
    
# Asynchronous function to process a single unit_id with ensemble reviewers
async def process_unit_id_with_ensemble(unit_id, waveform_plot_folder, model, example_ids,with_isi, with_qm):
    await asyncio.sleep(1)
    reviewers = [1, 2, 3]
    tasks = [classify_waveform_independently(unit_id, reviewer_id, waveform_plot_folder, model, example_ids, with_isi, with_qm) for reviewer_id in reviewers]
    reviews = await asyncio.gather(*tasks)

    # Aggregate results
    aggregated_result = aggregate_results(reviews, unit_id)
    return aggregated_result

async def run_in_batch(unit_ids,waveform_plot_folder, model, example_ids, with_isi, with_qm, n_semaphore):
    results = []
    semaphore = asyncio.Semaphore(n_semaphore)
    batch_size = 20
        
    async def limited_task(unit_id, progress_bar):
        async with semaphore:
            result = await process_unit_id_with_ensemble(unit_id, waveform_plot_folder, model, example_ids, with_isi, with_qm)
            progress_bar.update(1) 
            return result

    with tqdm(total=len(unit_ids), desc="Processing Units", unit="unit", file=sys.stdout, leave=True) as progress_bar:
        tasks = [limited_task(unit_id, progress_bar) for unit_id in unit_ids]
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            results.extend(await asyncio.gather(*batch))
            
    return results

def make_score_file(folder_name, score_folder):
    score_dfs = []
    columns = []
    base = Path(f'VLM_outputs_{folder_name}')
    for file in base.iterdir():
        result = pd.read_csv(file)
        score_dfs.append(result["average_score"])
        columns.append(file.name[7:-4])
    score_df = pd.concat(score_dfs, axis=1)
    score_df.columns = columns
    score_df.index = result['template_id']
    score_df.to_csv(f"{score_folder}/score_{folder_name}.csv")
    print(f"{', '.join(map(str, columns))} scores saved")
