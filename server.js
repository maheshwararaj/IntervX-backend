import express from 'express';
import cors from 'cors';
import multer from 'multer';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';
import * as tf from '@tensorflow/tfjs-node';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import dotenv from 'dotenv';
import { readFileSync } from 'fs';
import natural from 'natural';

dotenv.config();

const app = express();
const upload = multer({ dest: 'uploads/' });

import faiss from 'faiss-node';

import { pipeline } from '@xenova/transformers';
import * as tf from '@tensorflow/tfjs-node';
import { GPT2Tokenizer } from 'gpt2-tokenizer';

let encoder = null;
let nerPipeline = null;
let faissIndex = null;
let metadata = [];
let questionModel = null;
let tokenizer = null;
let currentContext = {};
let conversationHistory = [];

const VECTOR_DIM = 512; 

const ENTITY_MAPPING = {
    'PER': ['PERSON', 'PER'],
    'ORG': ['ORG', 'ORGANIZATION'],
    'LOC': ['LOC', 'GPE', 'LOCATION'],
    'DATE': ['DATE', 'TIME'],
    'SKILLS': ['SKILL', 'TECHNOLOGY'],
    'PROJECTS': ['PROJECT']
};

import { TECH_SKILLS } from './constants/skills.js';
import { PROJECT_INDICATORS } from './constants/projectIndicators.js';

async function initializeModels() {
    try {
        console.log('Loading models...');
        const [useModel, nerModel] = await Promise.all([
            use.load(),
            pipeline('ner', 'Xenova/bert-base-NER')
        ]);
        
        encoder = useModel;
        nerPipeline = nerModel;
        
        // Initialize GPT-2 tokenizer
        tokenizer = new GPT2Tokenizer();
        await tokenizer.init();
        
        // Load pre-trained GPT-2 model weights and architecture
        const modelPath = './models/gpt2-qa-model';
        try {
            questionModel = await tf.loadLayersModel(`file://${modelPath}/model.json`);
            console.log('Loaded existing fine-tuned model');
        } catch (e) {
            console.log('Creating new model...');
            questionModel = await createGPT2Model();
            await questionModel.save(`file://${modelPath}`);
        }
        
        faissIndex = new faiss.IndexFlatL2(VECTOR_DIM);
        console.log('Models and FAISS index loaded successfully');
    } catch (error) {
        console.error('Error loading models:', error);
        throw error;
    }
}

async function createGPT2Model() {
    const vocabSize = tokenizer.vocab_size;
    const model = tf.sequential();
    
    // Simple transformer-based architecture
    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: 256,
        inputLength: 512
    }));
    
    model.add(tf.layers.lstm({
        units: 512,
        returnSequences: true
    }));
    
    model.add(tf.layers.dense({
        units: vocabSize,
        activation: 'softmax'
    }));
    
    model.compile({
        optimizer: tf.train.adam(1e-4),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    return model;
}

initializeModels();

app.use(cors());
app.use(express.json());

const ENTITY_TYPES = {
    PER: 'Person',
    ORG: 'Organization',
    LOC: 'Location',
    DATE: 'Date'
};
async function extractTextFromPDF(pdfPath) {
    const dataBuffer = readFileSync(pdfPath);
    const data = await pdfParse(dataBuffer);
    
    const text = data.text
        .replace(/\s+/g, ' ')  // Replace multiple spaces with single space
        .replace(/[\r\n]+/g, '\n')  // Normalize line endings
        .trim();
    
    console.log('Extracted text:', text.substring(0, 500) + '...');
    return text;
}

async function extractRelevantSections(text) {
    if (!nerPipeline) {
        throw new Error('NER model not initialized');
    }

    console.log('Processing text...');
    console.log('Input text sample:', text.substring(0, 200));
    
    try {
        const sections = {};
        Object.keys(ENTITY_TYPES).forEach(type => {
            sections[type] = [];
        });

        const cleanText = text
            .replace(/\s+/g, ' ')
            .replace(/[\r\n]+/g, ' ')
            .trim();
        const sentences = cleanText.match(/[^.!?]+[.!?]+/g) || [cleanText];
        console.log('Running NER model...');
        const entities = await nerPipeline(cleanText, {
            aggregation_strategy: 'simple'
        });

        console.log('Raw NER output:', entities);

        for (const entity of entities) {
            const { entity_group, word } = entity;
            
            for (const [sectionType, validTypes] of Object.entries(ENTITY_MAPPING)) {
                if (validTypes.includes(entity_group)) {
                    if (!sections[sectionType]) {
                        sections[sectionType] = [];
                    }
                    sections[sectionType].push(word);
                    break;
                }
            }
        }

        sections.SKILLS = [];
        for (const skill of TECH_SKILLS) {
            const escapedSkill = skill.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const skillRegex = new RegExp(`\\b${escapedSkill}\\b`, 'gi');
            
            if (skillRegex.test(cleanText)) {
                sections.SKILLS.push(skill);
            }
        }

        sections.PROJECTS = [];
        for (const sentence of sentences) {
            const hasProjectIndicator = PROJECT_INDICATORS.some(indicator => 
                new RegExp(`\\b${indicator}\\b`, 'i').test(sentence)
            );

            if (hasProjectIndicator) {
                const projectInfo = sentence.trim();
                
                const cleanedProject = projectInfo
                    .replace(/^(project|developed|created|built|implemented):\s*/i, '')
                    .replace(/^[-•]\s*/, '');
                
                sections.PROJECTS.push(cleanedProject);
            }
        }


        console.log('Extracted sections:', sections);
        
        const result = {};
        for (const [type, values] of Object.entries(sections)) {
            result[type] = values.length > 0 ? [...new Set(values)].join(', ') : 'Not found';
        }
        
        console.log('Final result:', {skill:result.SKILLS,projects:result.PROJECTS});
        return result;
        
    } catch (error) {
        console.error('Error in NER processing:', error);
        throw error;
    }
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

async function generateQuestion(skills, projects, previousAnswer = null) {
    try {
        const context = [
            'Skills:', skills.join(', '),
            'Projects:', projects.join(', ')
        ].join('\n');

        if (previousAnswer) {
            context += '\nPrevious Answer: ' + previousAnswer;
        }

        const prompt = `Based on the following context, generate a relevant interview question:\n${context}\n\nQuestion:`;
        
        // Tokenize input
        const encoded = tokenizer.encode(prompt);
        const paddedInput = tf.tensor2d([encoded.padEnd(512, 0)]);
        
        // Generate output
        const predictions = questionModel.predict(paddedInput);
        const output = await predictions.argMax(-1).array();
        
        // Decode output tokens
        const question = tokenizer.decode(output[0]);
        return question.replace(prompt, '').trim();
    } catch (error) {
        console.error('Error generating question:', error);
        throw error;
    }
}

async function updateModel(question, answer, feedback) {
    try {
        const trainingData = `Question: ${question}\nAnswer: ${answer}\nFeedback: ${feedback}`;
        
        // Prepare training data
        const encoded = tokenizer.encode(trainingData);
        const paddedInput = tf.tensor2d([encoded.padEnd(512, 0)]);
        
        // Create target data (shifted by one position)
        const target = encoded.slice(1).concat([0]);
        const paddedTarget = tf.tensor2d([target.padEnd(512, 0)]);
        
        // Fine-tune the model
        await questionModel.fit(paddedInput, paddedTarget, {
            epochs: 1,
            batchSize: 1,
            verbose: 1
        });
        // Save updated model
        await questionModel.save('file://./models/gpt2-qa-model');
        console.log('Model updated with new training example');
    } catch (error) {
        console.error('Error updating model:', error);
        throw error;
    }
}

app.get('/', (req, res) => {
    res.json({ message: 'Welcome to InterX API' });
});

app.post('/upload', upload.single('resume'), async (req, res) => {
    try {

        const extractedText = await extractTextFromPDF(req.file.path);
        const relevantSections = await extractRelevantSections(extractedText);
        await storeEmbedding(relevantSections, req.file.originalname);
        res.json({ message: 'Resume data extracted and stored successfully!', data: relevantSections });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

const PORT = process.env.PORT || 5001;

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});