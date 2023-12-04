import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


def create_basic_features(train_logs_df, train_scores_df):
    group_actions = train_logs_df.groupby('id')['event_id'].sum().to_frame('event_count')
    train_df = train_scores_df.merge(group_actions, on='id')

    time_spend = train_logs_df.groupby('id')['up_time'].apply(np.ptp).to_frame('total_time').reset_index()
    train_df = train_df.merge(time_spend, on='id')

    word_count = train_logs_df.groupby('id')['word_count'].max().to_frame('word_count')
    train_df = train_df.merge(word_count, on='id')

    mean_action_time = train_logs_df.groupby('id')['action_time'].mean().to_frame('mean_action_time')
    train_df = train_df.merge(mean_action_time, on='id')

    train_df = train_df.drop('id', axis=1)
    train_df = train_df.astype('float32')

    train_df = train_df[:(int(len(train_df) * 0.8))]
    test_df = train_df[(int(len(train_df) * 0.8)):]

    train_features = train_df.to_numpy()[:, 1:]
    test_features = test_df.to_numpy()[:, 1:]

    # robust_scaler = RobustScaler(unit_variance=True)
    # robust_scaler_train_data = robust_scaler.fit_transform(train_features)
    # robust_scaler_test_data = robust_scaler.fit_transform(test_features)

    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_features)
    scaled_test_data = scaler.transform(test_features)

    scaled_train_data_features = scaled_train_data
    scaled_test_data_features = scaled_test_data

    train_ds = tf.data.Dataset.from_tensor_slices((scaled_train_data_features, train_df.score))

    test_ds = tf.data.Dataset.from_tensor_slices((scaled_test_data_features, train_df.score))

    train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds
