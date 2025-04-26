import logging
import numpy as np
from scipy.io.wavfile import write

from dotenv import load_dotenv
import os
from io import BytesIO

from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from csm import load_csm_1b
import torchaudio
import streamlit as st
from silero_vad import load_silero_vad
import whisper
from orpheus_cpp import OrpheusCpp
from pathlib import Path

load_dotenv()

logger = logging.getLogger("voice-assistant")
logger.setLevel(logging.INFO)

PERSIST_DIR = "./chat-engine-storage"

@st.cache_resource
def vad_model():
    silero_model = load_silero_vad()
    return silero_model

@st.cache_resource
def speaker_model():
    # stt = OrpheusCpp(lang="en")
    # return stt
    stt = load_csm_1b("cpu")
    return stt


st.title("Voice AI + LlamaIndex RAG")

st.write("## Upload your docs")
files = st.file_uploader(label="Your docs", accept_multiple_files=True)
if not files: exit()
for uploaded_file in files:
    bytes_data = uploaded_file.read()
    with open(Path("docs")/uploaded_file.name, "wb") as w:
        w.write(bytes_data)


st.write("## Record Audio input")

voice_file = st.audio_input(label="Audio Input")
result = st.chat_input("Your message")

# whisper_model = whisper.load_model("turbo")
llm = Ollama(model="gemma3", request_timeout=120.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

index = None


if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)



if True:
    if voice_file:
        voice_file = whisper.load_audio(voice_file)
        whisper_model = whisper.load_model("turbo")

        result = whisper_model.transcribe(voice_file)

        st.write("## Detected text")
        st.write(result)
    if result:
        query_engine = index.as_query_engine()
        # orpheus = speaker_model()
        q_resp = query_engine.query(result)

        speaker = speaker_model()
        audio = speaker.generate(text=str(q_resp), context=[], speaker=0, max_audio_length_ms=100_000)
        torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), speaker.sample_rate)
        st.audio("output.wav", autoplay=True)








