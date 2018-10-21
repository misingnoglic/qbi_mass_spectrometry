# Code to serve website

from scripts.biLSTM import BiLSTM, MultiheadAttention
from scripts.models import TestModel
from scripts.trainer import Trainer
import numpy as np
from flask import Flask, render_template, request, jsonify
from parse_data import amino_acid_name_parse
from calculate_frag_mz import get_frag_mz
app = Flask(__name__)

class Config:
    lr = 0.0001
    batch_size = 8
    max_epoch = 50
    gpu = False
    

neutral_loss_choices = [0, 17, 18, 35, 36, 44, 46]
n_neutral_losses = len(neutral_loss_choices)
n_charges = 7    

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/output_json')
def output_json():
    protein_name = request.args.get('protein_name')
    has_beginning_charge, name_one_hot_encoded, charge = amino_acid_name_parse(protein_name)
    mz_data, intensities = get_prediction(has_beginning_charge, name_one_hot_encoded, charge)
    d = {
        'protein_name': protein_name,
        'mz_data': mz_data,
        'intensities': intensities
    }
    return jsonify(d)


def get_prediction(has_beginning_charge, name_one_hot_encoded, charge):
    # Hardcoded for now until we can do actual ML
    
    opt=Config()
    net = TestModel(input_dim=24,
                    n_tasks=2*n_charges,
                    embedding_dim=256,
                    hidden_dim_lstm=128,
                    hidden_dim_attention=32,
                    n_lstm_layers=2,
                    n_attention_heads=8,
                    gpu=opt.gpu)
    trainer = Trainer(net, opt)
    
    trainer.load('./saved_models/model_flipped_bkp.pth')
    total_intensities = np.random.normal(70000, 20000)  
    X = np.array(name_one_hot_encoded)
    inputs = [(X, 
               (has_beginning_charge*1, charge), 
               np.zeros((X.shape[0]-1, 2*n_charges)))]
    pred = trainer.predict(inputs)[0]
    
    mz_data = []
    intensities = []
    peaks = np.where(pred > 0.005)
    for position, peak_type in zip(*peaks):
        b_y = (peak_type >= 7) * 1
        charge = peak_type - b_y * 7 + 1
        if b_y:
          ion_type = 'y'
          pos = (pred.shape[0] + 1) - position - 1
        else:
          ion_type = 'b'
          pos = position + 1
        mz_data.append(get_frag_mz(name_one_hot_encoded, pos, ion_type, charge))
        intensities.append(pred[position, peak_type] * total_intensities)
    return mz_data, intensities


if __name__ == '__main__':
    app.run(host='0.0.0.0')
