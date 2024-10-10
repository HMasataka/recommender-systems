from collections import defaultdict
import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
from typing import Dict, List
import dataclasses

np.random.seed(0)


@dataclasses.dataclass(frozen=True)
# 推薦システムの評価
class Metrics:
    rmse: float
    precision_at_k: float
    recall_at_k: float

    # 評価結果を出力する時に少数は第３桁までにする
    def __repr__(self):
        return f"rmse={self.rmse:.3f}, Precision@K={self.precision_at_k:.3f}, Recall@K={self.recall_at_k:.3f}"


@dataclasses.dataclass(frozen=True)
# 推薦システムの学習と評価に使うデータセット
class Dataset:
    # 学習用の評価値データセット
    train: pd.DataFrame
    # テスト用の評価値データセット
    test: pd.DataFrame
    # ランキング指標のテストデータセット。キーはユーザーID、バリューはユーザーが高評価したアイテムIDのリスト。
    test_user2items: Dict[int, List[int]]
    # アイテムのコンテンツ情報
    item_content: pd.DataFrame


@dataclasses.dataclass(frozen=True)
# 推薦システムの予測結果
class RecommendResult:
    # テストデータセットの予測評価値。RMSEの評価
    rating: pd.DataFrame
    # キーはユーザーID、バリューはおすすめアイテムIDのリスト。ランキング指標の評価。
    user2items: Dict[int, List[int]]


class MetricCalculator:
    def calc(
        self,
        true_rating: List[float],
        pred_rating: List[float],
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> Metrics:
        rmse = self._calc_rmse(true_rating, pred_rating)
        precision_at_k = self._calc_precision_at_k(true_user2items, pred_user2items, k)
        recall_at_k = self._calc_recall_at_k(true_user2items, pred_user2items, k)
        return Metrics(rmse, precision_at_k, recall_at_k)

    def _precision_at_k(
        self, true_items: List[int], pred_items: List[int], k: int
    ) -> float:
        if k == 0:
            return 0.0

        p_at_k = (len(set(true_items) & set(pred_items[:k]))) / k
        return p_at_k

    def _recall_at_k(
        self, true_items: List[int], pred_items: List[int], k: int
    ) -> float:
        if len(true_items) == 0 or k == 0:
            return 0.0

        r_at_k = (len(set(true_items) & set(pred_items[:k]))) / len(true_items)
        return r_at_k

    def _calc_rmse(self, true_rating: List[float], pred_rating: List[float]) -> float:
        return np.sqrt(mean_squared_error(true_rating, pred_rating))

    def _calc_recall_at_k(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> float:
        scores = []
        # テストデータに存在する各ユーザーのrecall@kを計算
        for user_id in true_user2items.keys():
            r_at_k = self._recall_at_k(
                true_user2items[user_id], pred_user2items[user_id], k
            )
            scores.append(r_at_k)
        return np.mean(scores)

    def _calc_precision_at_k(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> float:
        scores = []
        # テストデータに存在する各ユーザーのprecision@kを計算
        for user_id in true_user2items.keys():
            p_at_k = self._precision_at_k(
                true_user2items[user_id], pred_user2items[user_id], k
            )
            scores.append(p_at_k)
        return np.mean(scores)


class DataLoader:
    def __init__(
        self,
        num_users: int = 1000,
        num_test_items: int = 5,
        data_path: str = "ml-10M100K/",
    ):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path

    def load(self) -> Dataset:
        ratings, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(ratings)
        # ranking用の評価データは、各ユーザーの評価値が4以上の映画だけを正解とする
        # キーはユーザーID、バリューはユーザーが高評価したアイテムIDのリスト
        movielens_test_user2items = (
            movielens_test[movielens_test.rating >= 4]
            .groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        return Dataset(
            movielens_train, movielens_test, movielens_test_user2items, movie_content
        )

    def _split_data(self, movielens: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        # 学習用とテスト用にデータを分割する
        # 各ユーザの直近の５件の映画を評価用に使い、それ以外を学習用とする
        # まずは、それぞれのユーザが評価した映画の順序を計算する
        # 直近付与した映画から順番を付与していく(0始まり)
        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(
            ascending=False, method="first"
        )
        movielens_train = movielens[movielens["rating_order"] > self.num_test_items]
        movielens_test = movielens[movielens["rating_order"] <= self.num_test_items]
        return movielens_train, movielens_test

    def _load(self) -> (pd.DataFrame, pd.DataFrame):
        # 映画の情報の読み込み(10197作品)
        # movie_idとタイトル名のみ使用
        m_cols = ["movie_id", "title", "genre"]
        movies = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"),
            names=m_cols,
            sep="::",
            encoding="latin-1",
            engine="python",
        )
        # genreをlist形式で保持する
        movies["genre"] = movies.genre.apply(lambda x: x.split("|"))

        # ユーザが付与した映画のタグ情報の読み込み
        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        user_tagged_movies = pd.read_csv(
            os.path.join(self.data_path, "tags.dat"),
            names=t_cols,
            sep="::",
            engine="python",
        )
        # tagを小文字にする
        user_tagged_movies["tag"] = user_tagged_movies["tag"].str.lower()
        movie_tags = user_tagged_movies.groupby("movie_id").agg({"tag": list})

        # タグ情報を結合する
        movies = movies.merge(movie_tags, on="movie_id", how="left")

        # 評価データの読み込み
        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            os.path.join(self.data_path, "ratings.dat"),
            names=r_cols,
            sep="::",
            engine="python",
        )

        # user数をnum_usersに絞る
        valid_user_ids = sorted(ratings.user_id.unique())[: self.num_users]
        ratings = ratings[ratings.user_id <= max(valid_user_ids)]

        # 上記のデータを結合する
        movielens_ratings = ratings.merge(movies, on="movie_id")

        return movielens_ratings, movies


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass

    def run_sample(self) -> None:
        # Movielensのデータを取得
        movielens = DataLoader(
            num_users=1000, num_test_items=5, data_path="ml-10M100K/"
        ).load()
        # 推薦計算
        recommend_result = self.recommend(movielens)
        # 推薦結果の評価
        metrics = MetricCalculator().calc(
            movielens.test.rating.tolist(),
            recommend_result.rating.tolist(),
            movielens.test_user2items,
            recommend_result.user2items,
            k=10,
        )
        print(metrics)


class RandomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # ユーザーIDとアイテムIDに対して、０始まりのインデックスを割り振る
        unique_user_ids = sorted(dataset.train.user_id.unique())
        unique_movie_ids = sorted(dataset.train.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        # ユーザー×アイテムの行列で、各セルの予測評価値は0.5〜5.0の一様乱数とする
        pred_matrix = np.random.uniform(
            0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids))
        )

        # rmse評価用に、テストデータに出てくるユーザーとアイテムの予測評価値を格納する
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            # テストデータのアイテムIDが学習用に登場していない場合も乱数を格納する
            if row["movie_id"] not in movie_id2index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue
            # テストデータに現れるユーザーIDとアイテムIDのインデックスを取得し、評価値行列の値を取得する
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        # ランキング評価用のデータ作成
        # 各ユーザに対するおすすめ映画は、そのユーザがまだ評価していない映画の中からランダムに10作品とする
        # キーはユーザーIDで、バリューはおすすめのアイテムIDのリスト
        pred_user2items = defaultdict(list)
        # ユーザーがすでに評価した映画を取得する
        user_evaluated_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        for user_id in unique_user_ids:
            user_index = user_id2index[user_id]
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = unique_movie_ids[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break
        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    RandomRecommender().run_sample()
