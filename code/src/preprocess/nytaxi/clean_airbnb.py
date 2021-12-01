import os
import sys

import pandas as pd


def clean_airbnb(raw_airbnb_path, out_airbnb_path):
    raw_airbnb_data = pd.read_csv(raw_airbnb_path)

    # fill null value
    raw_airbnb_data.fillna({'reviews_per_month': 0}, inplace=True)
    raw_airbnb_data.fillna({'name': "null"}, inplace=True)
    raw_airbnb_data.fillna({'host_name': "null"}, inplace=True)
    raw_airbnb_data.fillna({'last_review': "null"}, inplace=True)

    assert (raw_airbnb_data.isnull().sum().to_numpy() == 0).all()

    # remove abnormal high prices larger than $1,000 per day
    raw_airbnb_data = raw_airbnb_data[raw_airbnb_data['price'] < 1000]

    # add the length of name as a feature
    raw_airbnb_data["name_length"] = raw_airbnb_data['name'].map(str).apply(len)

    # set all the minimum nights larger than 30 to 30
    raw_airbnb_data.loc[(raw_airbnb_data.minimum_nights > 30), 'minimum_nights'] = 30

    raw_airbnb_data.drop(columns=['id', 'host_id', 'host_name', 'name', 'last_review'], inplace=True)

    # set categorical to one-hot
    out_airbnb_data = pd.get_dummies(raw_airbnb_data,
                                     columns=['neighbourhood_group', 'neighbourhood', 'room_type'],
                                     prefix=['nhg', 'nh', 'rt'], drop_first=True)

    out_airbnb_data.to_csv(out_airbnb_path)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/nytaxi")  # change working directory
    clean_airbnb("AB_NYC_2019.csv", "airbnb_clean.csv")

