
#!/usr/bin/env python3
"""
Standalone RAG Q&A Application
Run this single file and it will:
1. Start the FastAPI backend server
2. Automatically open the frontend in your browser
3. Everything works seamlessly together
"""

import os
import io
import re
import logging
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Embeddings & Vector DB
from sentence_transformers import SentenceTransformer
import faiss

# Parsing
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

# Optional LLM
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    logger.warning("OpenAI not available")

# ------------------------- Configuration -------------------------

@dataclass
class Config:
    """Configuration settings for the RAG system."""
    max_file_size_mb: int = 200
    max_chunk_tokens: int = 1000
    min_chunk_tokens: int = 50
    default_chunk_tokens: int = 450
    default_overlap: int = 50
    max_context_chars: int = 8000
    embedding_batch_size: int = 32
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.2

config = Config()

# ------------------------- HTML Frontend Template -------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Q&A with LLM Comparison</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            min-height: 100vh;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 0;
        }

        .header h1 {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(45deg, #fff, #f0f8ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .header p {
            color: rgba(255,255,255,0.8);
            font-size: 1.2rem;
            font-weight: 300;
        }

        .main-content {
            display: grid;
            grid-template-columns: 300px 1fr 350px;
            gap: 30px;
            align-items: start;
        }

        .sidebar, .status-panel, .main-panel {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            position: sticky;
            top: 20px;
        }

        .sidebar h3, .status-panel h3 {
            font-size: 1.3rem;
            margin-bottom: 20px;
            color: #4338ca;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .config-group {
            margin-bottom: 25px;
        }

        .config-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #374151;
            font-size: 0.9rem;
        }

        .slider-container {
            position: relative;
            margin: 15px 0;
        }

        .slider {
            width: 100%;
            height: 6px;
            background: linear-gradient(90deg, #e5e7eb, #4338ca);
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
        }

        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #4338ca;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(67,56,202,0.3);
            transition: all 0.2s ease;
        }

        .slider::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            box-shadow: 0 4px 20px rgba(67,56,202,0.4);
        }

        .slider-value {
            position: absolute;
            top: -30px;
            right: 0;
            background: #4338ca;
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .select-wrapper {
            position: relative;
        }

        select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            background: white;
            font-size: 0.9rem;
            transition: all 0.2s ease;
            appearance: none;
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3e%3c/svg%3e");
            background-position: right 12px center;
            background-repeat: no-repeat;
            background-size: 16px;
            padding-right: 40px;
        }

        select:focus {
            border-color: #4338ca;
            box-shadow: 0 0 0 3px rgba(67,56,202,0.1);
            outline: none;
        }

        .api-key-section {
            background: #f8fafc;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }

        .api-key-section h4 {
            color: #374151;
            margin-bottom: 15px;
            font-size: 1rem;
        }

        .api-input {
            width: 100%;
            padding: 10px 12px;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            font-size: 0.9rem;
            font-family: monospace;
            background: white;
            transition: border-color 0.2s ease;
        }

        .api-input:focus {
            border-color: #4338ca;
            outline: none;
        }

        .main-panel {
            position: static;
        }

        .upload-section {
            margin-bottom: 40px;
        }

        .upload-section h2 {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            color: #374151;
            font-size: 1.5rem;
        }

        .upload-area {
            border: 2px dashed #cbd5e1;
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            background: linear-gradient(45deg, #f8fafc, #f1f5f9);
            position: relative;
            overflow: hidden;
        }

        .upload-area::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(67,56,202,0.05), transparent);
            transform: rotate(45deg);
            transition: all 0.6s ease;
            opacity: 0;
        }

        .upload-area:hover {
            border-color: #4338ca;
            background: linear-gradient(45deg, #faf5ff, #f3e8ff);
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(67,56,202,0.1);
        }

        .upload-area:hover::before {
            opacity: 1;
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }

        .upload-icon {
            font-size: 3rem;
            margin-bottom: 15px;
            opacity: 0.6;
        }

        .upload-text {
            color: #6b7280;
            margin-bottom: 5px;
        }

        .upload-subtext {
            color: #9ca3af;
            font-size: 0.9rem;
        }

        .browse-btn {
            background: linear-gradient(135deg, #4338ca, #6366f1);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }

        .browse-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(67,56,202,0.3);
        }

        .question-section h2 {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            color: #374151;
            font-size: 1.5rem;
        }

        .question-textarea {
            width: 100%;
            min-height: 120px;
            padding: 20px;
            border: 2px solid #e5e7eb;
            border-radius: 16px;
            font-size: 1rem;
            font-family: inherit;
            resize: vertical;
            transition: all 0.2s ease;
            background: rgba(255,255,255,0.8);
        }

        .question-textarea:focus {
            border-color: #4338ca;
            box-shadow: 0 0 0 4px rgba(67,56,202,0.1);
            outline: none;
            background: white;
        }

        .answer-mode {
            margin: 30px 0;
        }

        .answer-mode h3 {
            margin-bottom: 15px;
            color: #374151;
            font-size: 1.2rem;
        }

        .mode-buttons {
            display: flex;
            gap: 15px;
        }

        .mode-btn {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            font-weight: 500;
            position: relative;
            overflow: hidden;
        }

        .mode-btn.active {
            border-color: #4338ca;
            background: linear-gradient(135deg, #4338ca, #6366f1);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(67,56,202,0.2);
        }

        .mode-btn:not(.active):hover {
            border-color: #4338ca;
            background: #f8fafc;
            transform: translateY(-1px);
        }

        .submit-btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }

        .submit-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(16,185,129,0.3);
        }

        .submit-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            margin-bottom: 15px;
            border-radius: 12px;
            transition: all 0.3s ease;
        }

        .status-item.online {
            background: linear-gradient(135deg, #d1fae5, #a7f3d0);
            border-left: 4px solid #10b981;
        }

        .status-item.offline {
            background: linear-gradient(135deg, #fed7d7, #feb2b2);
            border-left: 4px solid #ef4444;
        }

        .status-item.warning {
            background: linear-gradient(135deg, #fef3cd, #fde68a);
            border-left: 4px solid #f59e0b;
        }

        .status-label {
            font-weight: 600;
            color: #374151;
        }

        .status-value {
            font-weight: 700;
            font-size: 1.1rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }

        .metric-card {
            background: rgba(255,255,255,0.7);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.3);
        }

        .metric-card:hover {
            background: rgba(255,255,255,0.9);
            transform: translateY(-2px);
        }

        .metric-number {
            font-size: 2rem;
            font-weight: 700;
            color: #4338ca;
            display: block;
        }

        .metric-label {
            color: #6b7280;
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 5px;
        }

        .answer-section {
            margin-top: 40px;
            padding: 30px;
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }

        .answer-section h3 {
            color: #374151;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .answer-content {
            background: #f8fafc;
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #4338ca;
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .sources-section {
            margin-top: 20px;
            padding: 20px;
            background: #fef7f0;
            border-radius: 12px;
            border-left: 4px solid #f97316;
        }

        .sources-section h4 {
            color: #ea580c;
            margin-bottom: 15px;
        }

        .source-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border: 1px solid #fed7aa;
        }

        .source-meta {
            font-size: 0.9rem;
            color: #9a3412;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .source-text {
            color: #451a03;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f4f6;
            border-radius: 50%;
            border-top-color: #4338ca;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .error-message, .success-message {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }

        .error-message {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-left: 4px solid #ef4444;
            color: #dc2626;
        }

        .success-message {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 4px solid #22c55e;
            color: #166534;
        }

        @media (max-width: 1200px) {
            .main-content {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .status-panel,
            .sidebar {
                position: static;
            }
        }

        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .mode-buttons {
                flex-direction: column;
            }
            
            .metric-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🧠 RAG Q&A with LLM Comparison</h1>
            <p>Intelligent document analysis powered by advanced language models</p>
        </header>

        <div class="main-content">
            <aside class="sidebar">
                <h3>⚙️ Configuration</h3>
                
                <div class="config-group">
                    <label>Chunk Size</label>
                    <div class="slider-container">
                        <input type="range" class="slider" id="chunkSize" min="100" max="1000" value="450">
                        <div class="slider-value" id="chunkSizeValue">450</div>
                    </div>
                </div>

                <div class="config-group">
                    <label>Overlap Tokens</label>
                    <div class="slider-container">
                        <input type="range" class="slider" id="overlapTokens" min="0" max="200" value="50">
                        <div class="slider-value" id="overlapValue">50</div>
                    </div>
                </div>

                <div class="config-group">
                    <label>Retrieval Count (k)</label>
                    <div class="slider-container">
                        <input type="range" class="slider" id="retrievalCount" min="1" max="20" value="5">
                        <div class="slider-value" id="retrievalValue">5</div>
                    </div>
                </div>

                <div class="config-group">
                    <label>Embedding Model</label>
                    <div class="select-wrapper">
                        <select id="embeddingModel">
                            <option value="sentence-transformers/all-MiniLM-L6-v2">all-MiniLM-L6-v2</option>
                            <option value="sentence-transformers/all-mpnet-base-v2">all-mpnet-base-v2</option>
                            <option value="text-embedding-ada-002">text-embedding-ada-002</option>
                        </select>
                    </div>
                </div>

                <div class="api-key-section">
                    <h4>🔐 LLM Configuration</h4>
                    <label>OpenAI API Key</label>
                    <input type="password" class="api-input" placeholder="sk-..." id="apiKey">
                    <div style="margin-top: 15px;">
                        <label>How to get your API key:</label>
                        <ol style="margin-top: 10px; padding-left: 20px; font-size: 0.85rem; color: #6b7280;">
                            <li>Visit <a href="https://platform.openai.com/login" target="_blank" style="color: #4338ca;">OpenAI account</a></li>
                            <li>Navigate to <a href="https://platform.openai.com/api-keys" target="_blank" style="color: #4338ca;">API keys section</a></li>
                            <li>Create new secret key</li>
                        </ol>
                    </div>
                </div>
            </aside>

            <main class="main-panel">
                <div class="upload-section">
                    <h2>📄 Document Upload</h2>
                    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                        <div class="upload-icon">📁</div>
                        <div class="upload-text">Drag and drop files here</div>
                        <div class="upload-subtext">Limit 200MB per file • PDF, MD, MARKDOWN, HTML, HTM</div>
                        <button class="browse-btn" type="button">Browse Files</button>
                        <input type="file" id="fileInput" multiple accept=".pdf,.md,.markdown,.html,.htm" style="display: none;">
                    </div>
                </div>

                <div class="question-section">
                    <h2>❓ Ask Your Question</h2>
                    <textarea class="question-textarea" placeholder="Enter your question here..." id="questionInput"></textarea>
                    
                    <div class="answer-mode">
                        <h3>Choose Answer Mode</h3>
                        <div class="mode-buttons">
                            <button class="mode-btn active" data-mode="documents">📚 Documents Only</button>
                            <button class="mode-btn" data-mode="chatgpt">🤖 ChatGPT Only</button>
                            <button class="mode-btn" data-mode="compare">⚖️ Compare Both</button>
                        </div>
                    </div>

                    <button class="submit-btn" id="submitBtn">✨ Get Answer</button>
                </div>

                <div id="answerSection" style="display: none;"></div>
            </main>

            <aside class="status-panel">
                <h3>📊 System Status</h3>
                
                <div class="status-item" id="llmStatus">
                    <span class="status-label">LLM Status</span>
                    <span class="status-value">● Checking...</span>
                </div>
                
                <div class="status-item" id="docsStatus">
                    <span class="status-label">Documents</span>
                    <span class="status-value">No Docs</span>
                </div>

                <div class="metric-grid">
                    <div class="metric-card">
                        <span class="metric-number" id="docsCount">0</span>
                        <div class="metric-label">Documents Loaded</div>
                    </div>
                    <div class="metric-card">
                        <span class="metric-number" id="chunksCount">0</span>
                        <div class="metric-label">Chunks Indexed</div>
                    </div>
                </div>

                <div style="margin-top: 25px; padding: 20px; background: rgba(79, 70, 229, 0.1); border-radius: 12px; border-left: 4px solid #4338ca;">
                    <h4 style="color: #4338ca; margin-bottom: 10px;">💡 Pro Tips</h4>
                    <ul style="color: #6b7280; font-size: 0.9rem; line-height: 1.5;">
                        <li>• Upload multiple documents for better context</li>
                        <li>• Ask specific questions for better results</li>
                        <li>• Try different chunk sizes for optimal performance</li>
                    </ul>
                </div>
            </aside>
        </div>
    </div>

    <script>
        const API_BASE = window.location.origin;
        
        let currentConfig = {
            chunkSize: 450,
            overlapTokens: 50,
            retrievalCount: 5,
            embeddingModel: 'sentence-transformers/all-MiniLM-L6-v2',
            apiKey: ''
        };

        document.addEventListener('DOMContentLoaded', function() {
            initializeSliders();
            setupEventListeners();
            updateStatus();
        });

        function initializeSliders() {
            const sliders = [
                { id: 'chunkSize', valueId: 'chunkSizeValue', configKey: 'chunkSize' },
                { id: 'overlapTokens', valueId: 'overlapValue', configKey: 'overlapTokens' },
                { id: 'retrievalCount', valueId: 'retrievalValue', configKey: 'retrievalCount' }
            ];

            sliders.forEach(slider => {
                const element = document.getElementById(slider.id);
                const valueElement = document.getElementById(slider.valueId);
                
                element.addEventListener('input', () => {
                    const value = parseInt(element.value);
                    valueElement.textContent = value;
                    currentConfig[slider.configKey] = value;
                    updateConfiguration();
                });
            });
        }

        function setupEventListeners() {
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                });
            });

            document.getElementById('fileInput').addEventListener('change', handleFileUpload);
            document.getElementById('submitBtn').addEventListener('click', handleSubmit);

            document.getElementById('apiKey').addEventListener('change', (e) => {
                if (e.target.value) {
                    currentConfig.apiKey = e.target.value;
                    updateConfiguration();
                }
            });

            document.getElementById('embeddingModel').addEventListener('change', (e) => {
                currentConfig.embeddingModel = e.target.value;
                updateConfiguration();
            });

            const uploadArea = document.querySelector('.upload-area');
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#4338ca';
            });

            uploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#cbd5e1';
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#cbd5e1';
                const files = Array.from(e.dataTransfer.files);
                handleFiles(files);
            });
        }

        async function updateConfiguration() {
            try {
                const response = await fetch(`${API_BASE}/configure`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        chunk_size: currentConfig.chunkSize,
                        overlap_tokens: currentConfig.overlapTokens,
                        retrieval_count: currentConfig.retrievalCount,
                        embedding_model: currentConfig.embeddingModel,
                        api_key: currentConfig.apiKey
                    })
                });
                if (response.ok) {
                    updateStatus();
                }
            } catch (error) {
                console.error('Error updating configuration:', error);
            }
        }

        async function updateStatus() {
            try {
                const response = await fetch(`${API_BASE}/status`);
                if (response.ok) {
                    const status = await response.json();
                    
                    const llmStatusElement = document.getElementById('llmStatus');
                    const llmValue = llmStatusElement.querySelector('.status-value');
                    if (status.llm_status === 'online') {
                        llmStatusElement.className = 'status-item online';
                        llmValue.innerHTML = '● Online';
                        llmValue.style.color = '#10b981';
                    } else {
                        llmStatusElement.className = 'status-item offline';
                        llmValue.innerHTML = '● Offline';
                        llmValue.style.color = '#ef4444';
                    }

                    const docsStatusElement = document.getElementById('docsStatus');
                    const docsValue = docsStatusElement.querySelector('.status-value');
                    if (status.documents_status === 'ready') {
                        docsStatusElement.className = 'status-item online';
                        docsValue.innerHTML = '● Ready';
                        docsValue.style.color = '#10b981';
                    } else {
                        docsStatusElement.className = 'status-item warning';
                        docsValue.innerHTML = 'No Docs';
                        docsValue.style.color = '#f59e0b';
                    }

                    document.getElementById('docsCount').textContent = status.documents_loaded;
                    document.getElementById('chunksCount').textContent = status.chunks_indexed;
                }
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }

        async function handleFileUpload(event) {
            const files = Array.from(event.target.files);
            await handleFiles(files);
        }

        async function handleFiles(files) {
            if (files.length === 0) return;

            const formData = new FormData();
            files.forEach(file => formData.append('files', file));

            const submitBtn = document.getElementById('submitBtn');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = '📤 Uploading...';
            submitBtn.disabled = true;

            try {
                const response = await fetch(`${API_BASE}/upload`, {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    showSuccessMessage(`Successfully processed ${result.processed_files.length} files with ${result.new_chunks} chunks`);
                    
                    document.querySelector('.upload-text').textContent = `${files.length} file(s) processed`;
                    document.querySelector('.upload-subtext').textContent = files.map(f => f.name).join(', ');
                    updateStatus();
                } else {
                    const error = await response.json();
                    showErrorMessage(`Upload failed: ${error.detail}`);
                }
            } catch (error) {
                showErrorMessage(`Upload error: ${error.message}`);
            } finally {
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            }
        }

        async function handleSubmit() {
            const question = document.getElementById('questionInput').value.trim();
            const activeMode = document.querySelector('.mode-btn.active').dataset.mode;
            
            if (!question) {
                showErrorMessage('Please enter a question!');
                return;
            }

            const submitBtn = document.getElementById('submitBtn');
            const originalText = submitBtn.textContent;
            submitBtn.innerHTML = '<span class="loading-spinner"></span> Processing...';
            submitBtn.disabled = true;

            try {
                const response = await fetch(`${API_BASE}/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: question,
                        mode: activeMode,
                        top_k: currentConfig.retrievalCount
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    displayAnswers(result, activeMode);
                } else {
                    const error = await response.json();
                    showErrorMessage(`Query failed: ${error.detail}`);
                }
            } catch (error) {
                showErrorMessage(`Network error: ${error.message}`);
            } finally {
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            }
        }

        function displayAnswers(result, mode) {
            const answerSection = document.getElementById('answerSection');
            let html = '<div class="answer-section">';

            if (mode === 'documents') {
                html += '<h3>📄 Answer from Your Documents</h3>';
                html += `<div class="answer-content">${result.documents_answer || 'No answer found.'}</div>`;
                
                if (result.sources && result.sources.length > 0) {
                    html += '<div class="sources-section">';
                    html += '<h4>📚 Retrieved Sources</h4>';
                    result.sources.forEach((source, index) => {
                        html += `<div class="source-item">`;
                        html += `<div class="source-meta">Source ${index + 1} (Relevance: ${source.relevance.toFixed(2)}) - ${source.metadata.source}`;
                        if (source.metadata.page) {
                            html += `, Page ${source.metadata.page}`;
                        }
                        html += '</div>';
                        html += `<div class="source-text">${source.text}</div>`;
                        html += '</div>';
                    });
                    html += '</div>';
                }
            } else if (mode === 'chatgpt') {
                html += '<h3>🤖 ChatGPT Answer</h3>';
                html += `<div class="answer-content">${result.chatgpt_answer || 'No answer from ChatGPT.'}</div>`;
            } else if (mode === 'compare') {
                html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">';
                
                html += '<div>';
                html += '<h3>📄 Your Documents Say:</h3>';
                html += `<div class="answer-content">${result.documents_answer || 'No relevant documents found.'}</div>`;
                html += '</div>';
                
                html += '<div>';
                html += '<h3>🤖 ChatGPT Says:</h3>';
                html += `<div class="answer-content">${result.chatgpt_answer || 'No answer from ChatGPT.'}</div>`;
                html += '</div>';
                
                html += '</div>';

                if (result.sources && result.sources.length > 0) {
                    html += '<div class="sources-section">';
                    html += '<h4>📚 Retrieved Sources</h4>';
                    result.sources.forEach((source, index) => {
                        html += `<div class="source-item">`;
                        html += `<div class="source-meta">Source ${index + 1} (Relevance: ${source.relevance.toFixed(2)}) - ${source.metadata.source}`;
                        if (source.metadata.page) {
                            html += `, Page ${source.metadata.page}`;
                        }
                        html += '</div>';
                        html += `<div class="source-text">${source.text}</div>`;
                        html += '</div>';
                    });
                    html += '</div>';
                }
            }

            html += '</div>';
            answerSection.innerHTML = html;
            answerSection.style.display = 'block';
            answerSection.scrollIntoView({ behavior: 'smooth' });
        }

        function showErrorMessage(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = message;
            
            const mainPanel = document.querySelector('.main-panel');
            mainPanel.appendChild(errorDiv);
            
            setTimeout(() => errorDiv.remove(), 5000);
        }

        function showSuccessMessage(message) {
            const successDiv = document.createElement('div');
            successDiv.className = 'success-message';
            successDiv.textContent = message;
            
            const mainPanel = document.querySelector('.main-panel');
            mainPanel.appendChild(successDiv);
            
            setTimeout(() => successDiv.remove(), 5000);
        }

        setInterval(updateStatus, 30000);
    </script>
</body>
</html>
"""

# ------------------------- Pydantic Models -------------------------

class QueryRequest(BaseModel):
    question: str
    mode: str = "documents"
    top_k: int = 5

class ConfigRequest(BaseModel):
    chunk_size: Optional[int] = None
    overlap_tokens: Optional[int] = None
    retrieval_count: Optional[int] = None
    embedding_model: Optional[str] = None
    api_key: Optional[str] = None

class StatusResponse(BaseModel):
    llm_status: str
    documents_status: str
    documents_loaded: int
    chunks_indexed: int
    loaded_sources: List[str]

# ------------------------- Backend Code (same as before) -------------------------

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]

def validate_file_size(file_bytes: bytes, filename: str) -> bool:
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > config.max_file_size_mb:
        logger.error(f"File {filename} is too large ({size_mb:.1f}MB). Max size: {config.max_file_size_mb}MB")
        return False
    return True

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace('\u00a0', ' ').replace('\xa0', ' ')
    s = re.sub(r"\s+", " ", s).strip()
    return "" if len(s) < 10 else s

def read_pdf(file_bytes: bytes, source_name: str) -> List[Chunk]:
    chunks = []
    try:
        if not validate_file_size(file_bytes, source_name):
            return []
        
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        for i, page in enumerate(doc):
            try:
                text = clean_text(page.get_text("text") or "")
                if text:
                    chunks.append(Chunk(
                        text=text, 
                        metadata={"source": source_name, "page": i + 1, "type": "pdf"}
                    ))
            except Exception as e:
                logger.warning(f"Error processing page {i+1} of {source_name}: {e}")
        
        doc.close()
        logger.info(f"Successfully extracted {len(chunks)} pages from {source_name}")
    except Exception as e:
        logger.error(f"Failed to read PDF {source_name}: {e}")
    return chunks

def read_html(raw: str, source_name: str) -> List[Chunk]:
    try:
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style", "noscript", "iframe", "object", "embed"]):
            tag.decompose()
        
        text = clean_text(soup.get_text(" "))
        if not text:
            return []
        
        return [Chunk(text=text, metadata={"source": source_name, "type": "html"})]
    except Exception as e:
        logger.error(f"Failed to read HTML {source_name}: {e}")
        return []

def read_markdown(raw: str, source_name: str) -> List[Chunk]:
    try:
        html = MarkdownIt().render(raw)
        return read_html(html, source_name)
    except Exception as e:
        logger.error(f"Failed to read Markdown {source_name}: {e}")
        return []

def chunk_text_semantic(text: str, chunk_tokens: int = 450, overlap: int = 50) -> List[str]:
    if not text or not text.strip():
        return []

    chunk_tokens = max(config.min_chunk_tokens, min(chunk_tokens, config.max_chunk_tokens))
    overlap = max(0, min(overlap, chunk_tokens // 2))
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(sentence.split()) > chunk_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            words = sentence.split()
            for i in range(0, len(words), chunk_tokens - overlap):
                chunks.append(" ".join(words[i:i + chunk_tokens]))
            continue

        if len((current_chunk + " " + sentence).split()) > chunk_tokens and current_chunk:
            chunks.append(current_chunk)
            
            if overlap > 0:
                overlap_words = current_chunk.split()[-overlap:]
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence
            
    if current_chunk:
        chunks.append(current_chunk)
    
    return [chunk for chunk in chunks if len(chunk.split()) >= 5]

class VectorIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.embedder = SentenceTransformer(model_name)
            self.dim = self.embedder.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(self.dim)
            self.chunks: List[Chunk] = []
            self.model_name = model_name
            logger.info(f"Initialized VectorIndex with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorIndex: {e}")
            raise

    def _norm(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-12)

    def add_chunks(self, chunks: List[Chunk]) -> bool:
        if not chunks:
            logger.warning("No chunks to add")
            return False
            
        try:
            texts = [c.text for c in chunks if c.text.strip()]
            if not texts:
                logger.warning("No valid text content in chunks")
                return False
            
            encode_args = {
                'batch_size': config.embedding_batch_size,
                'convert_to_numpy': True, 
                'show_progress_bar': False
            }
            
            vecs = self.embedder.encode(texts, **encode_args)
            vecs = self._norm(vecs)
            self.index.add(vecs)
            self.chunks.extend([c for c in chunks if c.text.strip()])
            
            logger.info(f"Added {len(texts)} chunks to index")
            return True
        except Exception as e:
            logger.error(f"Failed to add chunks to index: {e}")
            return False

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        if not query.strip() or self.index.ntotal == 0:
            return []
            
        try:
            q = self._norm(self.embedder.encode([query], convert_to_numpy=True))
            D, I = self.index.search(q, min(k, self.index.ntotal))
            
            return [
                (self.chunks[idx], float(score)) 
                for score, idx in zip(D[0], I[0]) 
                if idx != -1
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def clear(self):
        try:
            self.index.reset()
            self.chunks.clear()
            logger.info("Index cleared")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")

# Global state
class AppState:
    def __init__(self):
        self.vindex: Optional[VectorIndex] = None
        self.ingested_sources: List[str] = []
        self.chunk_tokens = config.default_chunk_tokens
        self.overlap = config.default_overlap

app_state = AppState()

# RAG Pipeline
SYS_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer cannot be found in the context, clearly state that you don't know. "
    "Always cite your sources using the format [Source: filename, page X] when applicable. "
    "Be concise but comprehensive in your answers."
)

def build_prompt(question: str, contexts: List[Tuple[Chunk, float]], max_ctx_chars: int = config.max_context_chars) -> str:
    blocks = []
    used_chars = 0
    
    for chunk, _ in contexts:
        meta = chunk.metadata
        cite_parts = [meta.get('source')]
        if 'page' in meta:
            cite_parts.append(f"p.{meta['page']}")
        
        citation = ", ".join(filter(None, cite_parts)) or "Unknown source"
        block = f"[Source: {citation}]\n{chunk.text}\n"
        
        if used_chars + len(block) > max_ctx_chars and blocks:
            break
            
        blocks.append(block)
        used_chars += len(block)
    
    context = "\n---\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

def generate_answer(question: str, contexts: List[Tuple[Chunk, float]]) -> str:
    if not contexts:
        return "I don't have any relevant information to answer this question."
    
    prompt = build_prompt(question, contexts)
    
    if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=config.openai_model,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
    
    response_parts = ["*Fallback Answer (No LLM):*\n\nBased on the most relevant document snippets:\n"]
    for i, (chunk, score) in enumerate(contexts[:3], 1):
        response_parts.append(f"*Snippet {i} (Relevance: {score:.2f})*\n> {chunk.text[:300]}...\n")
    return "\n".join(response_parts)

def get_chatgpt_answer(question: str) -> Optional[str]:
    if not (_HAS_OPENAI and os.getenv("OPENAI_API_KEY")):
        return None

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            temperature=config.temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"ChatGPT API call failed: {e}")
        return None

# FastAPI Application
app = FastAPI(title="RAG Q&A Standalone App", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTML_TEMPLATE

@app.get("/status", response_model=StatusResponse)
async def get_status():
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    llm_status = "online" if has_api_key else "offline"
    
    docs_loaded = len(app_state.ingested_sources)
    chunks_indexed = app_state.vindex.index.ntotal if app_state.vindex else 0
    docs_status = "ready" if docs_loaded > 0 else "no_docs"
    
    return StatusResponse(
        llm_status=llm_status,
        documents_status=docs_status,
        documents_loaded=docs_loaded,
        chunks_indexed=chunks_indexed,
        loaded_sources=app_state.ingested_sources
    )

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not app_state.vindex:
        try:
            app_state.vindex = VectorIndex()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize vector index: {str(e)}")
    
    processed_files = []
    total_new_chunks = 0
    
    for file in files:
        if file.filename in app_state.ingested_sources:
            continue
        
        try:
            file_bytes = await file.read()
            chunks = []
            
            suffix = file.filename.split(".")[-1].lower()
            
            if suffix == "pdf":
                chunks = read_pdf(file_bytes, file.filename)
            else:
                raw_content = file_bytes.decode("utf-8", errors="ignore")
                if suffix in ("md", "markdown"):
                    chunks = read_markdown(raw_content, file.filename)
                elif suffix in ("html", "htm"):
                    chunks = read_html(raw_content, file.filename)
            
            if chunks:
                fine_chunks = []
                for chunk in chunks:
                    fine_chunks.extend([
                        Chunk(text, chunk.metadata) 
                        for text in chunk_text_semantic(chunk.text, app_state.chunk_tokens, app_state.overlap)
                    ])
                
                if app_state.vindex.add_chunks(fine_chunks):
                    app_state.ingested_sources.append(file.filename)
                    processed_files.append(file.filename)
                    total_new_chunks += len(fine_chunks)
                    
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
    
    return {
        "message": f"Successfully processed {len(processed_files)} files",
        "processed_files": processed_files,
        "new_chunks": total_new_chunks,
        "total_files": len(app_state.ingested_sources),
        "total_chunks": app_state.vindex.index.ntotal if app_state.vindex else 0
    }

@app.post("/query")
async def query_documents(request: QueryRequest):
    if not app_state.vindex:
        raise HTTPException(status_code=400, detail="No vector index initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    response_data = {"question": request.question, "mode": request.mode}
    
    if request.mode in ["documents", "compare"]:
        results = app_state.vindex.search(request.question, k=request.top_k)
        
        if results:
            rag_answer = generate_answer(request.question, results)
            sources = [
                {
                    "text": chunk.text[:300] + "...",
                    "metadata": chunk.metadata,
                    "relevance": float(score)
                }
                for chunk, score in results
            ]
            response_data["documents_answer"] = rag_answer
            response_data["sources"] = sources
        else:
            response_data["documents_answer"] = "No relevant documents found for your query."
            response_data["sources"] = []
    
    if request.mode in ["chatgpt", "compare"]:
        chatgpt_answer = get_chatgpt_answer(request.question)
        if chatgpt_answer:
            response_data["chatgpt_answer"] = chatgpt_answer
        else:
            response_data["chatgpt_answer"] = "Failed to get response from ChatGPT. Please check your API key."
    
    return response_data

@app.post("/configure")
async def update_configuration(request: ConfigRequest):
    updated_fields = []
    
    if request.chunk_size is not None:
        app_state.chunk_tokens = max(config.min_chunk_tokens, min(request.chunk_size, config.max_chunk_tokens))
        updated_fields.append("chunk_size")
    
    if request.overlap_tokens is not None:
        app_state.overlap = max(0, min(request.overlap_tokens, app_state.chunk_tokens // 2))
        updated_fields.append("overlap_tokens")
    
    if request.api_key is not None and request.api_key.strip():
        os.environ["OPENAI_API_KEY"] = request.api_key
        updated_fields.append("api_key")
    
    # Only reinitialize vector index if embedding model is explicitly changed
    if request.embedding_model is not None and request.embedding_model != (app_state.vindex.model_name if app_state.vindex else None):
        try:
            app_state.vindex = VectorIndex(model_name=request.embedding_model)
            app_state.ingested_sources = []  # Clear sources only when model changes
            updated_fields.append("embedding_model")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load embedding model: {str(e)}")
    
    return {
        "message": "Configuration updated successfully",
        "updated_fields": updated_fields,
        "current_config": {
            "chunk_size": app_state.chunk_tokens,
            "overlap_tokens": app_state.overlap,
            "has_api_key": bool(os.getenv("OPENAI_API_KEY")),
            "embedding_model": app_state.vindex.model_name if app_state.vindex else None
        }
    }

@app.post("/clear")
async def clear_index():
    if app_state.vindex:
        app_state.vindex.clear()
    app_state.ingested_sources = []
    return {"message": "Index cleared successfully"}

def start_server():
    """Start the FastAPI server"""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

def open_browser():
    """Open the browser after a short delay"""
    import time
    time.sleep(2)  # Wait for server to start
    webbrowser.open("http://localhost:8000")
    print("\n🚀 RAG Q&A Application is now running!")
    print("📱 Frontend: http://localhost:8000")
    print("📡 API Docs: http://localhost:8000/docs")
    print("\n🛑 Press Ctrl+C to stop the server")

if __name__ == "__main__":
    print("🔥 Starting RAG Q&A Application...")
    print("📦 Loading dependencies...")
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the server (this blocks)
    start_server()
