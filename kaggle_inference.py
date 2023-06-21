from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import pandas as pd
import numpy as np
import torch
import os
import glob

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


class DeplotWithClassificationHead(torch.nn.Module):
    def __init__(self, deplot_weigths_path):
        super().__init__()
        
        self.deplot = Pix2StructForConditionalGeneration.from_pretrained(deplot_weigths_path)
        self.classifiaction_head = torch.nn.Linear(768, 5)
        
        
    def forward(self, flattened_patches, attention_mask, labels=None, decoder_input_ids=None, decoder_attention_mask=None):
        deplot_output = self.deplot.forward(flattened_patches=flattened_patches, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)
        encoder_mean = deplot_output["encoder_last_hidden_state"].mean(1)
        deplot_output["type_logits"] = self.classifiaction_head(encoder_mean)

        return deplot_output
    
    @torch.inference_mode()
    def generate(self, flattened_patches, attention_mask, max_new_tokens=128):
        output = self.deplot.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, return_dict_in_generate=True, output_hidden_states=True, max_new_tokens=max_new_tokens)
        
        sequences = output["sequences"]
        types = self.classifiaction_head(output["encoder_hidden_states"][-1].mean(1)).argmax(1).cpu().numpy().tolist()
        
        return sequences, types
    
TYPE2INT = {
    "scatter": 0, 
    "line": 1, 
    "vertical_bar": 2, 
    "dot": 3, 
    "horizontal_bar": 4
}

INT2TYPE = {v: k for k, v in TYPE2INT.items()}

@dataclass
class DataCollatorCTCWithPadding:

    processor: Pix2StructProcessor

    def __call__(self, batch):
        
        new_batch = {"flattened_patches":[], "attention_mask":[], "id":[]}
        
        for item in batch:
            new_batch["flattened_patches"].append(item["flattened_patches"])
            new_batch["attention_mask"].append(item["attention_mask"])
            new_batch["id"].append(item["id"])
            
        # replace padding with -100 to ignore loss correctly
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])

        return new_batch

class BenetechDataset(Dataset):
    
    def __init__(self, dataset_df, processor, mode="train_val"):
        super().__init__()
        
        assert mode in ("train_val", "inference"), "mode must be train_val or infernce"
        self.mode = mode
        
        self.processor = processor
        self.dataset_df = dataset_df

        self.pretokenized = "tokenized_transcript" in self.dataset_df.columns

    def __len__(self):
        return len(self.dataset_df)
    
    def tokenize_text(self, text):
        return np.squeeze(self.processor.tokenizer.encode(text, \
            padding="longest", return_tensors="pt", add_special_tokens=True, max_length=128))
        
    def __getitem__(self, idx):
        image_path = self.dataset_df.at[idx, "image_path"]
        image_id = os.path.basename(image_path).replace(".jpg", "")
        image = Image.open(image_path)
        
        encoding = self.processor(images=image, text="Generate underlying data table of the figure below:", add_special_tokens=True, return_tensors="pt", max_patches=512)
               
        output_dict = {
            "flattened_patches": encoding["flattened_patches"][0],
            "attention_mask": encoding["attention_mask"][0],
            "id": image_id}

        if self.mode == "train_val":
            
            target_text = self.dataset_df.at[idx, "target_text"]

            labels = self.dataset_df.at[idx, "tokenized_transcript"] if self.pretokenized \
                else self.tokenize_text(f"{target_text}")

            output_dict["target_text"] = target_text
            
            output_dict["target_text_x"] = self.dataset_df.at[idx, "target_text_x"]
            output_dict["target_text_y"] = self.dataset_df.at[idx, "target_text_y"]
            
            output_dict["x_axis_type"] = self.dataset_df.at[idx, "x_axis_type"]
            output_dict["y_axis_type"] = self.dataset_df.at[idx, "y_axis_type"]
            
            output_dict["labels"] = labels
            output_dict["target_type"] = torch.tensor(TYPE2INT[self.dataset_df.at[idx, "target_type"]], dtype=torch.long)

        return output_dict 
    
def process_batch(batch, device="cpu"):
    flattened_patches = batch["flattened_patches"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    img_id = batch["id"]
    
    return flattened_patches, attention_mask, img_id
    

if __name__ == "__main__":
    TEST_IMG_DIR = "/home/pawel/Projects/benetech-making-graphs-accessible/data/test/images"
    STATE_DICT_PATH = ""
    df = pd.DataFrame()
    df["image_path"] = glob.glob(f"{TEST_IMG_DIR}/*.jpg")

    processor = Pix2StructProcessor.from_pretrained(f"google/matcha-base")

    dataset = BenetechDataset(df, processor, mode="inference")

    data_collator = DataCollatorCTCWithPadding(processor)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)

    model = DeplotWithClassificationHead("deplot-model")

    state_dict = torch.load("/home/pawel/Projects/benetech-making-graphs-accessible/checkpoints/config10/last/pytorch_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    total_steps = len(dataloader)


    def process_output(output, types, ids):
        if isinstance(output, str):
            output = [output]

        outputs = []

        for text, tp, id_ in zip(output, types, ids):
        
            chart_type = INT2TYPE[tp]
        
            outputs.append([f"{id_}_x", ";".join([xy.split("|")[0].strip() for xy in text.split("<0x0A>")[1:] if "|" in xy]), f"{chart_type}"])
            outputs.append([f"{id_}_y", ";".join([xy.split("|")[1].strip() for xy in text.split("<0x0A>")[1:] if "|" in xy]), f"{chart_type}"])
        

        return outputs


    full_predictions = []

    with tqdm(enumerate(dataloader, 1), unit="batch", bar_format='{l_bar}{bar:10}{r_bar}', total=total_steps, position=0, leave=True) as progress_bar:
        progress_bar.set_description(f"Inference")

        for step, batch in progress_bar:
        
            flattened_patches, attention_mask, img_id = process_batch(batch)

            with torch.cuda.amp.autocast(enabled=False):
                predictions, types = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_new_tokens=256)
                predictions = processor.batch_decode(predictions, skip_special_tokens=True)
                predictions = ["<0x0A>".join(pred.split("<0x0A>")[1:]) for pred in predictions]
                
                full_predictions.extend(process_output(predictions, types, img_id))

        submission_df = pd.DataFrame(full_predictions, columns=["id", "data_series", "chart_type"])
        submission_df.to_csv("submission.csv", index=None)
            