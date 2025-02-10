import os
import base64
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Annotated, TypedDict, Literal, Tuple, List
from dotenv import load_dotenv
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import ToolNode
from prompt import system_prompt
from langgraph.types import Command
from textwrap import dedent
import streamlit as st
from util_gemini import display_message, render_conversation_history, get_conversation_summary
# Directory Setup
plot_dir = os.path.join(os.path.dirname(__file__), "tmp/plots")
os.makedirs(plot_dir, exist_ok=True)
load_dotenv()

@tool
def environment_setup_tool(is_neuropixel:bool, raw_data_path:str, processed_path:str, folder_name:str) -> str:
    """
    Sets up the environment by importing modules and retrieving initial parameters.
    This tool:
    - if user wants default, no inputs.
    - necesarry input ->is_neruopixel:bool
    - 'raw_data_path':str, 'processed_path':str, 'folder_name':str
    - Provides every folder path needed for saving data.

    - folder_name is the name of the folder that contains raw recording data.
    - processed_path is an empty folder where results will be saved.
    - raw_data_path is the parent folder path of folder_name. folder_name should be located in this path.
    - is_neruopixel is a bool type that indicates if this data is from Neuropixels or not. True for Neuropixels data.

    After execution, `folder_paths` and `params` will be available.
    """
    code = f"""
    import Agent.default as ad
    import Agent.utils as au
    import Agent.read as ar
    import Agent.process as ap
    import Agent.visualize as av

    params = ad.get_default_params({is_neuropixel})
    params['raw_data_path'] = {raw_data_path}
    params['processed_path'] = {processed_path}
    params['folder_name'] = {folder_name}
    folder_paths = ad.Folder_paths(params)
    # Please use python_repl_tool to execute this code directly.
    """
    return dedent(code)


@tool
def read_recording_tool(filter_frequency_minimum:float,filter_frequency_maximum:float,filter_operator:str, figure_height:float, probe_zoomin:bool) -> str:
    """
    Read recording and plot some infos.
    This tool:
    - if user wants default, no inputs.
    - 'filter_frequency_minimum': int, 'filter_frequency_maximum':int, 'filter_operator':str, 'figure_height':int, 'probe_zoomin':bool}
    - all inputs are for updating 'params'
    - Reads raw data and creates a recording. 
    - Shows the structure of the probe and compares the recording before and after filtering.
    
    - figure_height: is for Neuropixel data that determines the height of the probe plot. 50 is usually sufficient.
    - (filter_frequency_minimum, filter_frequency_maximum): is usually set to (300, 3000) or more. 
    - filter_operator : specifies the reference type for the filter. It is usually 'average' for mesh types and 'median' for Neuropixel types.
    - probe_zoomin: boolean whether to zoom in the probe or not. True recommend for neuropixel type.

    After execution, `recording`, `day_length`, and `recording_cmr` will be available.
    """
    code = f"""
    params['filter_frequency_minimum'] = {filter_frequency_minimum}
    params['filter_frequency_maximum'] = {filter_frequency_maximum}
    params['filter_operator'] = {filter_operator}
    params['figure_height'] = {figure_height}
    params['probe_zoomin'] = {probe_zoomin}
    
    recording, day_length = ar.read_data_folder(params, folder_paths)
    mesh_probe = au.create_probe(recording, params)
    av.show_probe(recording, params)
    recording_cmr = au.filters(recording, params)
    av.show_recording(recording, recording_cmr)
    # Please use python_repl_tool to execute this code directly.
    """
    return dedent(code)


@tool
def run_sorting_tool(sorter_name:str,load_if_exist:bool,modify_sorting_param:str|float|bool) -> str:
    """
    Executes the sorting algorithm and extracts waveforms.
    This tool:
    - if user wants default, no inputs.
    - sorter_name:str, load_if_exist:bool, modify_sorting_param:dict
    - sorter_name, load_if_exist is for updating 'params' and modify_sorting_param for 'get_sorting_params' function
    - key for modify_sorting_param can be obtained from ad.get_sorting_params(sorter_name, recording_cmr)
    
    - if user wants to see what kind of sorting params are avaliale or to see default sorting params for this sorter_name,
    execute code until 'ad.getting_sorting_params' line and it will print present parameters.
    - Gets sorting parameters and runs the sorter according to the parameters.
    - Extracts the waveform extractor.
    
    - sorter_name list: 'mountainsort4', 'mountainsort5', 'kilosort4', 'spykingcircus2', 'tridesclous2', 'tridesclous', 'herdingspikes'
    - load_if_exists: Boolean. When True, loads the file if it exists; otherwise, runs the process.
    After execution, `sorting` and `we` will be available.
    
    """
    code = f"""
    params['curated'] = False
    params['sorter_name'] = {sorter_name}
    params['load_if_exist'] = {load_if_exist}
    
    folder_paths.update(params['sorter_name'])
    sorting_params = ad.get_sorting_params(params['sorter_name'], recording_cmr, **{modify_sorting_param})
    sorting = ap.run_sorter(recording_cmr, params, sorting_params, folder_paths)
    we = ap.get_we(recording_cmr, sorting, params, folder_paths)
    # Please use python_repl_tool to execute this code directly.
    """
    return dedent(code)


@tool
def initial_visualization_tool(save_figures:bool,colored_template:bool,n_columns:int,figure_height:float) -> str:
    """
    Generates initial visualizations for quality checking.
    This tool:
    - if user wants default, no inputs.
    - 'save_figures':boolean, 'colored_template':boolean, 'n_columns':int, 'figure_height':int
    - all inputs for updating 'params'
    - Shows rasters, templates, unit waveforms, and ISI rate.
    - Users can choose whether to save, color the graph, and set the number of columns for each plot.
    - This visualization is performed before curation.
    
    - save_figures: bool types that decide whether to save the figure or not for each visualization.
    - n_columns: specifies the number of columns for figures. A value between 4 and 12 is recommended.
    - colored_template: is a bool type. True for a colored figure, False for a black-and-white figure.
    - figure_height: sets the height of the probe templates' figure. A value of 50 is sufficient.

    After execution, initial visualizations are generated.
    """
    code = f"""
    params['curated'] = False
    params['save_figures'] = {save_figures}
    params['colored_template'] = {colored_template}
    params['n_columns'] = {n_columns}
    params['figure_height'] = {figure_height}
    
    av.show_rasters(sorting, day_length, params, folder_paths)
    av.plot_templates(we, params, folder_paths)
    av.sorting_unit_show(we, params, folder_paths)
    av.plot_isi(sorting, params)
    #av.save_waveform(we, params, folder_paths)
    #av.save_all_in_one(we, params, folder_paths)
    # Please use python_repl_tool to execute this code directly.
    """
    return dedent(code)


@tool
def curation_merging_tool(min_snr:float,max_isi:float,show_metrics:bool,load_if_exist:bool) -> str:
    """
    Performs automated curation on the sorted units.
    This tool:
    - if user wants default, no inputs.
    - 'min_snr':float, 'max_isi':float, 'show_metrics':boolean, 'load_if_exist': boolean
    - all inputs for updating 'params'
    
    
    - Automatically curates based on thresholds; min_snr, max_isi
    - After curation, it provides new sorting and waveform extractor.
    - if show_metrics is true, it will show whoel metrics with curation values
    - keep_unit_ids is curated result. It show which units are regareded as real neuron according to max_isi and min_snr.
    
    - max_isi: Values between 0.5 and 1.5 are recommended.
    - min_snr: Values between 4.0 and 10.0 are recommended.
    - show_metrics: Boolean. Determines whether to show the matrix or not.
    - load_if_exists: Boolean. When True, loads the file if it exists; otherwise, runs the process again.
    When False, reruns merging regardless of saved files.  

    After execution, `keep_unit_ids`, `sorting_curated`, and `we_curated` will be available.
    """
    code = f"""
    params['curated'] = True
    params['min_snr'] = {min_snr}
    params['max_isi'] = {max_isi}
    params['load_if_exist'] = {load_if_exist}
    params['show_metrics'] = {show_metrics}
    
    keep_unit_ids, metrics = au.auto_curation(we, params)
    av.summary(metrics,we,keep_unit_ids,params,folder_paths)
    sorting_curated, we_curated = ap.units_merge(recording_cmr, sorting, keep_unit_ids, params, folder_paths, merge_unit_ids_pack = [])

    # Please use python_repl_tool to execute this code directly.
    """
    return dedent(code)


@tool
def final_visualization_tool(save_figures:bool,colored_template:bool,n_columns:int,figure_height:float) -> str:
    """
    Generates final visualizations.
    This tool:
    - if user wants default, no inputs.
    - 'save_figures':boolean, 'colored_template':boolean, 'n_columns':int, 'figure_height':int
    - all inputs for updating 'params'
    - Performs visualizations after curation.
    - Saves extremum channels as a CSV file.
    
    - save_figures: bool types that decide whether to save the figure or not for each visualization.
    - n_columns: specifies the number of columns for figures. A value between 4 and 12 is recommended.
    - colored_template: is a bool type. True for a colored figure, False for a black-and-white figure.
    - figure_height: sets the height of the probe templates' figure. A value of 50 is sufficient.

    After execution, final visualizations are generated and extremum channels will be saved.
    """
    code = f"""
    params['curated'] = True
    params['curated'] = False
    params['save_figures'] = {save_figures}
    params['colored_template'] = {colored_template}
    params['n_columns'] = {n_columns}
    params['figure_height'] = {figure_height}
    
    av.show_rasters(sorting_curated, day_length, params, folder_paths)
    av.plot_templates(we_curated, params, folder_paths)
    av.sorting_unit_show(we_curated, params, folder_paths)
    av.plot_isi(sorting_curated, params)
    au.save_extremum(we_curated, folder_paths)
    # Please use python_repl_tool to execute this code directly.
    
    """
    return dedent(code)


python_repl = PythonREPL()

@tool(response_format="content_and_artifact")
def python_repl_tool(query: str) -> Tuple[str, List[str]]:
    """A Python shell. Use this to execute python commands. Input should be a valid python command.
    The input query should be some code that can be directly executed by exec(query), you can do ```python```
    If you want to see the output of a value, you should print it out with `print(...)`. """
    
    plot_paths = []  # List to store file paths of generated plots
    result_summary = {}  # Initialize result summary dict for output
    result_parts = []  # List to store different parts of the output
    
    # Initialize Gemini model for code translation
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",temperature=0,max_tokens=None,timeout=None,max_retries=2)
    
    try:
        # Use Gemini to translate query to executable Python code if needed
        messages = [
            (
                "system",
                "You are a helpful assistant that translates natural language or code-like text into valid Python code. "
                "The code should be executable and not include any markdown formatting. "
                "Only output the actual executable Python code, nothing else."
            ),
            ("human", query)
        ]
        
        translated_code = llm.invoke(messages).content
        result_summary["translated_code"] = translated_code
        result_parts.append(f"Translated code:\n{translated_code}")
        
        # Execute the translated Python code with PythonREPL
        output = python_repl.run(translated_code)
        if output and output.strip():  # If there's printed output
            result_summary["output"] = output
            result_parts.append(f"Output:\n{output.strip()}")
        
        # After executing, check if any matplotlib figures were created
        figures = [plt.figure(i) for i in plt.get_fignums()]
        if figures:
            for fig in figures:
                # Generate filename
                plot_filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                # Create relative path
                rel_path = os.path.join("tmp/plots", plot_filename)
                # Convert to absolute path for saving
                abs_path = os.path.join(os.path.dirname(__file__), rel_path)
                
                fig.savefig(abs_path)
                plot_paths.append(rel_path)  # Store relative path
            plt.close("all")
            result_parts.append(f"Generated {len(plot_paths)} plot(s).")
            result_summary["status"] = f"Executed code and generated {len(plot_paths)} plot(s)."
        
        if not result_parts:  # If no output and no figures
            result_parts.append("Executed code successfully with no output. If you want to see the output of a value, you should print it out with `print(...)`.")
            result_summary["status"] = "Executed code successfully with no output."

    except Exception as e:
        error_message = f"Error executing code: {e}"
        result_parts.append(error_message)
        result_summary["status"] = error_message
        result_summary["output"] = None
        result_summary["translated_code"] = None
    
    # Join all parts of the result with newlines and add to summary
    result_summary["full_output"] = "\n\n".join(result_parts)
    
    # Return both the summary dict and plot paths (if any)
    return json.dumps(result_summary), plot_paths

# Tools List and Node Setup
tools = [
    python_repl_tool,
    environment_setup_tool,
    read_recording_tool,
    run_sorting_tool,
    initial_visualization_tool,
    curation_merging_tool,
    final_visualization_tool
]
tool_node = ToolNode(tools)

# Graph Setup
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    input_messages_len: list[int]
graph = StateGraph(GraphsState)

# Initialize Gemini models
gemini_2_0_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0).bind_tools(tools)
gemini_1_5_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0).bind_tools(tools)
gemini_1_5_flash_8b = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0).bind_tools(tools)
gemini_1_5_pro = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0).bind_tools(tools)

models = {
    "gemini-2.0-flash-exp": gemini_2_0_flash,
    "gemini-1.5-flash": gemini_1_5_flash,
    "gemini-1.5-flash-8b": gemini_1_5_flash_8b,
    "gemini-1.5-pro": gemini_1_5_pro
}

def _call_model(state: GraphsState, config: RunnableConfig) -> Command[Literal["tools", "__end__"]]:
    st.session_state["final_state"]["messages"] = state["messages"]
    model_name = config["configurable"].get("model", "gemini-2.0-flash-exp")  # default model
    llm = models[model_name]
    previous_message_count = len(state["messages"])
    state["input_messages_len"].append(previous_message_count)
    render_conversation_history(state["messages"][state["input_messages_len"][-2]:state["input_messages_len"][-1]])
    cur_messages_len = len(state["messages"])-state["input_messages_len"][0]  
    
    if cur_messages_len > 50:
        st.markdown(
        f"""
        <p style="color:blue; font-size:16px;">
            Current recursion step is {cur_messages_len}. Terminated because you exceeded the limit of 50.
        </p>
        """,
        unsafe_allow_html=True
        )
        st.session_state["render_last_message"] = False
        return Command(
            update={"messages": []},
            goto="__end__",
        )
    last_message = state["messages"][-1]
    # Check if last message is a ToolMessage and has artifacts
    if isinstance(last_message, ToolMessage) and hasattr(last_message, "artifact") and last_message.artifact and model_name != "gpt-3.5-turbo":
        content_list = [{
            "type": "text",
            "text": """
                Please analyze these generated images by the code above. Your tasks are to:
                1. Examine each visualization carefully
                2. Provide a detailed description of what you observe
                3. Assess the quality of the results

                If you determine the results are good:
                • Describe your observations in detail
                • Proceed directly to the next step without asking for user feedback

                If you determine improvements are needed:
                • Identify the specific issues
                • Modify the code to address these issues
                • Execute the updated code
            """
        }]
        
        # Add all PNG images to the content list
        for rel_path in last_message.artifact:
            if rel_path.endswith(".png"):
                # Convert relative path to absolute based on current script location
                abs_path = os.path.join(os.path.dirname(__file__), rel_path)
                if os.path.exists(abs_path):
                    with open(abs_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    })
        
        # Create a single message with all images if we found any
        if len(content_list) > 1:  # Only if we have at least one image
            image_message = HumanMessage(content=content_list,name="image_assistant")
            state["messages"].append(image_message)
    response = llm.invoke(state["messages"])
    if response.tool_calls:
        return Command(
            update={"messages": [response]},
            goto="tools",
        )
    else:
        st.session_state["render_last_message"] = True
        return Command(
            update={"messages": [response]},
            goto="__end__",
        )

graph.add_edge(START, "modelNode")
graph.add_node("tools", tool_node)
graph.add_node("modelNode", _call_model)
graph.add_edge("tools", "modelNode")
graph_runnable = graph.compile()

def invoke_our_graph(messages,model_choose):
    config = {"recursion_limit": 100, "configurable": {"model": model_choose}}
    return graph_runnable.invoke({"messages": messages,"input_messages_len":[len(messages)]},config=config)
