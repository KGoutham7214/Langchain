import { OpenAI } from "langchain/llms";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import * as dotenv from "dotenv";
import express from "express";
import cors from "cors";
const app = express();
app.use(express.json());
app.use(cors());
dotenv.config();


const textFile = "data";
const txtPath = `/${textFile}.txt`;
const VECTOR_STORE_PATH = `./${textFile}.index`;

async function runWithEmbeddings(question){
    const model = new OpenAI({});
    let vectorStore;
    if (fs.existsSync(VECTOR_STORE_PATH)) {
        console.log("exists vector store");
        vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
    }else{
        const text = fs.readFileSync(txtPath, "utf8");
        const textSplitter = new RecursiveCharacterTextSplitter({chunkSize: 1000});
        const docs = await textSplitter.createDocuments(text);
        vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
        await vectorStore.save(VECTOR_STORE_PATH);
    }

    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
    const res = await chain.call({
        query: question,
    });
    console.log("run done");
    return res;
    // console.log({res});
    
};

app.get("/ping", (req, res) => {
    res.json({
      message: "pong",
    });
  });

app.post("/", async (req,res)=>{
    const question = req.body.question;
    const response = await runWithEmbeddings(question);
    console.log(response);

    async function change(response){
      console.log( response );
      return await response?.text;
    }
    change(response).then((answer) => {
      console.log({ answer });
      const array = answer
        ?.split("\n")
        .filter((value) => value)
        .map((value) => value.trim());

      return array;
    })
    .then((answer) => {
      res.json({
        answer: answer,
        propt: question,
      });
    });
    
});

app.listen(3000,()=>{
    console.log("Server is running on port 3000");
});