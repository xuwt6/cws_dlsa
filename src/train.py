from src.model import dy_train_model

if __name__ == "__main__":
    dy_train_model(
        folder_path='../data',
        folder_path_utf8='../data_utf8',
        data_name='msr',
        max_epochs=20,
        batch_size=20,
        char_dims=50,
        word_dims=50,
        nhiddens=50,
        dropout_rate=0.2,
        max_word_len=50,
        margin_loss_discount=0.2,
        shuffle_data=True,
        lr=0.2,
        momentum=0.2,
        word_proportion=0.9,
        threshold_length=7,
        threshold_occurence=2
    )
