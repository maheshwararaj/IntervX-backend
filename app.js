import express from 'express';
import cors from 'cors';
import multer from 'multer';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';
import { readFileSync } from 'fs';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { pipeline } from "@xenova/transformers";
import faiss from 'faiss-node';
import { extract_prompt } from './embed.js';
import { first_question_prompt, followup_question_prompt } from './embed.js';

let encoder = null;
const faissIndex = new faiss.IndexFlatL2(768); 
const metadata = [];
let userData;
initializeEncoder();

async function initializeEncoder() {
    encoder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    console.log("Embedding model initialized");
}

const app = express();
const PORT = 3000;

app.use(express.json());
app.use(cors());
const upload = multer({ dest: 'uploads/' });

const apikey = "AIzaSyCgxmn8jRTWvcxCerow2nzexkw_38hlA5o";
const genAI = new GoogleGenerativeAI(apikey);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });



async function extractTextFromPDF(pdfPath) {
    const dataBuffer = readFileSync(pdfPath);
    const data = await pdfParse(dataBuffer);
    return data.text.replace(/\s+/g, ' ').trim();
}

async function storeEmbedding(extractedData, userId) {
    if (!encoder || !faissIndex) {
        throw new Error("Encoder or FAISS index not initialized");
    }

    const text = JSON.stringify(extractedData);
    
    const embedding = await encoder(text, { pooling: "mean", normalize: true });
    const vector = embedding.data;
    console.log(vector);
    // faissIndex.add([vector]);
    const vectorId = faissIndex.ntotal() - 1;
    metadata.push({ userId, data: extractedData, vector_id: vectorId });

    console.log(`Vector stored in FAISS successfully for user ${userId}`);

}

function getStoredEmbeddings() {
    return metadata.map((item) => {
        const vector = new Float32Array(768);
        faissIndex.reconstruct(item.vector_id, vector);
        return {
            ...item,
            vector: Array.from(vector)
        };
    });
}

app.get('/embeddings', (req, res) => {
    try {
        const embeddings = getStoredEmbeddings();
        res.json({ embeddings });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});


app.post('/upload', upload.single('resume'), async (req, res) => {
    try {
        const extractedText = await extractTextFromPDF(req.file.path);
        const prompt = extract_prompt + extractedText;
        console.log(extractedText);
        
        const response = await model.generateContent(prompt);
        let result = response.response.text();

        if (result.startsWith("```json")) {
            result = result.substring(7);
        }
        if (result.endsWith("```")) {
            result = result.slice(0, -3);
        }
        result = result.trim();

        const parsedData = JSON.parse(result);
        userData = parsedData;
        console.log(parsedData);
        console.log(req.body.userId)
        await storeEmbedding(parsedData, req.body.userId);

        res.json({ message: 'Resume data extracted and stored successfully!', data: parsedData });
    } catch (error) {
        res.status(500).json({ error: error.message , data: "hello" });
    }
});

app.get('/firstquestion', async (req, res) => {
    try {
        const prompt = first_question_prompt + JSON.stringify(userData);
        
        const response = await model.generateContent(prompt);
        let result = response.response.text();
        console.log(result)
        res.json({ message: 'First question generated successfully!', data: result });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/followupquestion', async (req, res) => {
    console.log(req.body)
    try {
        const prompt = followup_question_prompt + JSON.stringify(userData) + ". previous answer : "+req.body.userAnswer;
        console.log(prompt);
        const response = await model.generateContent(prompt);
        let result = response.response.text();
        console.log(result)
        res.json({ message: 'Follow up question generated successfully!', data: result });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});