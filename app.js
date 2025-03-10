import express from 'express';
import cors from 'cors';
import multer from 'multer';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';
import { readFileSync } from 'fs';
import { GoogleGenerativeAI } from '@google/generative-ai';

const app = express();
const PORT = 3000;

// Middleware to parse JSON requests
app.use(express.json());
app.use(cors());
const upload = multer({ dest: 'uploads/' });


const apikey = "AIzaSyCgxmn8jRTWvcxCerow2nzexkw_38hlA5o"
const genAI = new GoogleGenerativeAI(apikey);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });


// Basic route
app.get('/', async (req, res) => {
    const prompt = "difference between python and java";
    const response = await model.generateContent(prompt);
    const result = response.response.text();
    console.log(result)
    res.send(result);

});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});




async function extractTextFromPDF(pdfPath) {
    
    const dataBuffer = readFileSync(pdfPath);
    const data = await pdfParse(dataBuffer);
    
    const text = data.text
        .replace(/\s+/g, ' ')  
        .replace(/[\r\n]+/g, '\n') 
        .trim();
    
    console.log('Extracted text:', text.substring(0, 500) + '...');
    return text;
}

async function storeEmbedding(extractedData, fileName) {
    if (!encoder || !faissIndex) {
        throw new Error('Encoder or FAISS index not initialized');
    }

    const text = JSON.stringify(extractedData);
    const embedding = await encoder.embed([text]);
    const vector = await embedding.array();
    
    faissIndex.add(vector[0]);
    const vectorId = faissIndex.ntotal() - 1;
    metadata.push({
        fileName,
        data: extractedData,
        vector_id: vectorId
    });
    
    console.log(`Vector stored in FAISS successfully for ${fileName}`);
    console.log('Stored vector:', vector[0])
    return metadata[metadata.length - 1];
}

app.post('/upload', upload.single('resume'), async (req, res) => {
    console.log("hello")
    try {
        const extractedText = await extractTextFromPDF(req.file.path);
        console.log(extractedText);
        const prompt = `Extract the name,education, skills, projects, work experience and organization from the following text into a JSON format. Skills should be a string array. Only include technical skills related to programming languages, frameworks, and databases. Do not include achievements, or other skills. For projects, only include the name and description. Do not include the tech stack within the project object. organization means only companies not the universities or college,schools.The text is: ${extractedText}`;
        const response = await model.generateContent(prompt);
        let result = response.response.text();

        if (result.startsWith("```json")) {
            result = result.substring(7); // Remove "```json"
          }
          if (result.endsWith("```")) {
            result = result.slice(0, -3); // Remove "```"
          }
        
          // Remove leading/trailing whitespace and newlines
          result = result.trim();
        
        console.log("Result",result);
        res.json({ message: 'Resume data extracted and stored successfully!', data: JSON.parse(result) });

    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});