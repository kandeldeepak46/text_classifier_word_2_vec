### Run the Flask Server
```bash
python app.py
```
or
```bash
$ chmod +x experiments/run_server.sh
$ ./experiments/run_server.sh

```

### Train the Word2Vec model using
```bash
curl POST 'http://localhost:8080/api/v1/wv-model-training'  --form 'file=data/train_40k.csv'
```

### Train the LightBGM classifier usring
```bash
curl POST http://localhost:8080/api/v1/text-classifier-training \ --form 'file={file_location}' \
--form 'model_id="{model_id_which_is_returned_from_the_first_method}"'
```
### Test the model using
```bash
curl POST http://localhost:8080/api/v1/predict-text \
--header "Content-Type: application/json" \
--data '{"text": "{any_review_you_want_to_classify}",
"model_id": "{model_id_which_is_returned_from_the_first_method}"}'

```

### TODO
Docker Implementation