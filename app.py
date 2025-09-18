from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from collections import defaultdict
import random
import numpy as np
from typing import List, Dict  # Added this import
app = Flask(__name__, template_folder="templates")

# --- Dữ liệu và Mô hình ---
try:
    # 1. Tải tất cả sách và mô hình AI
    print("Loading data and model...")
    books_df_full = pd.read_csv(r"data\books.csv")
    ratings_df = pd.read_csv(r"data\ratings.csv")
    
    model_path = r"model\knn_basic_best-para_20k.pkl"
    with open(model_path, 'rb') as file:
        model_dict = pickle.load(file)
        model = model_dict["algo"]
        if not hasattr(model, 'predict'):
            raise TypeError("Loaded object is not a valid Surprise model with a 'predict' method")

    # 2. LẤY DANH SÁCH SÁCH MÀ MÔ HÌNH THỰC SỰ BIẾT
    print("Filtering books to match model's training data...")
    all_known_inner_ids = model.trainset.all_items()
    known_raw_ids = {int(model.trainset.to_raw_iid(inner_id)) for inner_id in all_known_inner_ids}
    
    # 3. LỌC DATAFRAME `books_df` ĐỂ CHỈ GIỮ LẠI NHỮNG SÁCH MÀ MÔ HÌNH BIẾT
    original_count = len(books_df_full)
    books_df = books_df_full[books_df_full['book_id'].isin(known_raw_ids)].copy()
    filtered_count = len(books_df)
    print(f"Finished filtering. Kept {filtered_count} of {original_count} books.")
    if filtered_count == 0:
        raise RuntimeError("Fatal: No books from books.csv match the model's training data.")

    # Chuyển đổi book_id sang chuỗi để khớp với trainset của Surprise
    books_df['book_id_str'] = books_df['book_id'].astype(str)

except FileNotFoundError as e:
    raise RuntimeError(f"Không tìm thấy tệp cần thiết: {e.filename}")
except Exception as e:
    raise RuntimeError(f"Lỗi khi tải dữ liệu hoặc mô hình: {e}")

# --- Hàm hỗ trợ ---
def get_book_info(book_id: int):
    """Trả về dict thông tin sách (hoặc None nếu không tìm thấy)."""
    book = books_df[books_df['book_id'] == book_id]
    if book.empty:
        return None
    return book.iloc[0].to_dict()

def _fill_with_global_avg_from_ratings(ratings_df: pd.DataFrame, books_df: pd.DataFrame, exclude_ids: set, need: int) -> List[Dict]:
    """Fill with books having highest average ratings from ratings.csv, excluding given IDs."""
    agg = ratings_df.groupby('book_id').agg(mean_rating=('rating', 'mean'), rating_count=('rating', 'count')).reset_index()
    agg = agg[agg['book_id'].isin(known_raw_ids)]  # Only known books
    agg = agg[~agg['book_id'].isin(exclude_ids)]
    cols_keep = [c for c in ['book_id', 'title', 'authors', 'image_url', 'average_rating', 'ratings_count'] if c in books_df.columns]
    info = books_df[cols_keep]
    merged = agg.merge(info, on='book_id', how='left')
    merged = merged.sort_values(['mean_rating', 'rating_count'], ascending=[False, False])
    top = merged.head(need)
    out = []
    for _, row in top.iterrows():
        out.append({
            'book_id': int(row['book_id']),
            'title': row.get('title'),
            'authors': row.get('authors'),
            'image_url': row.get('image_url'),
            'average_rating': row.get('average_rating', None),
            'ratings_count': row.get('ratings_count', None),
            'estimated_rating': round(float(row['mean_rating']), 2)
        })
    return out

# --- API Endpoints ---
@app.route("/")
def index():
    return render_template("new.html")

@app.route("/popular_books", methods=["GET"])
def popular_books():
    top_50 = books_df.sort_values("ratings_count", ascending=False).head(50)
    sample_size = min(10, len(top_50))
    if sample_size == 0:
        return jsonify([])
    random_10 = top_50.sample(n=sample_size)
    result = random_10[["book_id", "title", "image_url"]].to_dict(orient="records")
    return jsonify(result)

@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    query = request.args.get("query", "").lower()
    if not query:
        return jsonify([])
    starts_with = books_df[books_df["title"].str.lower().str.startswith(query)]
    contains = books_df[books_df["title"].str.lower().str.contains(query) & ~books_df.index.isin(starts_with.index)]
    suggestions = pd.concat([starts_with, contains]).head(10)
    return jsonify(suggestions[["book_id", "title"]].to_dict(orient="records"))

@app.route("/recommend", methods=["POST"])
def recommend_books():
    """
    Đề xuất sách dựa trên top 5 người dùng tương tự.
    """
    try:
        # 1. Lấy và xác thực đánh giá của người dùng
        user_ratings = {}
        rated_book_ids = set()

        user_ratings_by_book_id_raw = defaultdict(dict)
        for key, value in request.form.items():
            if key.startswith('book_id_'):
                idx = key.split('_')[-1]
                user_ratings_by_book_id_raw[idx]['book_id'] = int(value)
            elif key.startswith('rating_'):
                idx = key.split('_')[-1]
                user_ratings_by_book_id_raw[idx]['rating'] = int(value)

        for data in user_ratings_by_book_id_raw.values():
            bid = data.get('book_id')
            r = data.get('rating')
            if bid and r:
                # Only consider books known by the model for similarity calculation
                if bid in known_raw_ids:
                    user_ratings[bid] = float(r)
                    rated_book_ids.add(bid)

        if not user_ratings:
            # Fallback to global top 5 by average rating if no user ratings
            fallback = _fill_with_global_avg_from_ratings(ratings_df, books_df, rated_book_ids, need=5)
            return jsonify({'recommendations': fallback})

        user_id = 'new_user'
        trainset = model.trainset # Get the trainset from the loaded model

        # 2. Tìm top 5 user tương tự bằng KNNBasic
        similarities = []
        for uid in trainset.all_users():
            raw_uid = trainset.to_raw_uid(uid)
            sim_score = 0
            count = 0
            
            # Calculate similarity based on shared rated items
            for rated_bid, rated_r in user_ratings.items():
                try:
                    # Get the actual rating of the current 'uid' for 'rated_bid'
                    # You would need a way to get user ratings from the trainset
                    # trainset.ur is structured as {inner_uid: [(inner_iid, rating), ...]}
                    inner_rated_bid = trainset.to_inner_iid(rated_bid)
                    
                    # Find if 'uid' has rated 'rated_bid'
                    user_rated_items = {item_inner_id: rating for item_inner_id, rating in trainset.ur[uid]}
                    
                    if inner_rated_bid in user_rated_items:
                        actual_rating = user_rated_items[inner_rated_bid]
                        sim_score += (rated_r - actual_rating) ** 2  # Sum of squared differences
                        count += 1
                except ValueError: # book_id not in trainset
                    continue
            
            if count > 0:
                # Convert MSE-like score to similarity. Smaller MSE = higher similarity
                # Using 1 / (1 + RMSE) as a simple similarity metric
                sim_score = 1 / (1 + np.sqrt(sim_score / count)) 
                similarities.append((raw_uid, sim_score))
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_users = [uid for uid, _ in similarities[:5]]

        if not top_users:
            # Fallback if no similar users
            fallback = _fill_with_global_avg_from_ratings(ratings_df, books_df, rated_book_ids, need=5)
            return jsonify({'recommendations': fallback})

        # 3. Lấy các sách được đánh giá cao từ top 5 người dùng tương tự
        candidate_books = defaultdict(list)
        for uid in top_users:
            inner_uid = trainset.to_inner_uid(uid)
            # Get all ratings of this user from trainset
            user_all_ratings = [(trainset.to_raw_iid(iid), rating) 
                                for (iid, rating) in trainset.ur[inner_uid]]
            
            # Consider books with a rating above a certain threshold (e.g., 4.0 or 3.5)
            # and not already rated by the new user.
            for bid, rating in user_all_ratings:
                if bid not in rated_book_ids and rating >= 3.5:
                    candidate_books[bid].append(rating)

        if not candidate_books:
            # Fallback if no candidates
            fallback = _fill_with_global_avg_from_ratings(ratings_df, books_df, rated_book_ids, need=5)
            return jsonify({'recommendations': fallback})

        # 4. Sắp xếp các sách ứng cử viên dựa trên điểm trung bình của chúng từ người dùng tương tự
        # and then by ratings_count for tie-breaking
        ranked_candidates = []
        for bid, ratings_list in candidate_books.items():
            if ratings_list:
                avg_rating = np.mean(ratings_list)
                # Get ratings_count for tie-breaking
                book_info = books_df[books_df['book_id'] == bid]
                ratings_count = book_info['ratings_count'].iloc[0] if not book_info.empty else 0
                ranked_candidates.append((bid, avg_rating, ratings_count))
        
        # Sort by average rating (desc), then by ratings_count (desc)
        ranked_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # 5. Chọn top recommendations from similar users (up to 5)
        top_recs_from_similar = ranked_candidates[:5]
        
        recommendations = []
        chosen_ids = set()
        for bid, est, _ in top_recs_from_similar:
            info = get_book_info(bid)
            if info:
                info_clean = {k: '' if pd.isna(v) else v for k, v in info.items()}
                info_clean['estimated_rating'] = round(est, 2) # Use the average rating from similar users as estimated rating
                recommendations.append(info_clean)
                chosen_ids.add(bid)

        # If not enough, fill with global high average rating books, excluding chosen and rated
        deficit = 5 - len(recommendations)
        if deficit > 0:
            exclude_all = rated_book_ids | chosen_ids
            fillers = _fill_with_global_avg_from_ratings(ratings_df, books_df, exclude_all, need=deficit)
            recommendations.extend(fillers)
            chosen_ids.update([f['book_id'] for f in fillers])

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        print(f"💥 Lỗi server: {e}")
        return jsonify({'error': f'Đã xảy ra lỗi máy chủ: {e}'}), 500

if __name__ == "__main__":
    app.run(debug=True)


# ==================== CUSTOM RECOMMENDER (5 nearest users -> 25 liked books -> 5 picks) ====================
from typing import List, Dict, Optional, Tuple
import os

def _load_books_and_ratings():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    books_path = os.path.join(base_dir, "data", "books.csv")
    ratings_path = os.path.join(base_dir, "data", "ratings.csv")
    books_df = pd.read_csv(books_path)
    ratings_df = pd.read_csv(ratings_path)
    return books_df, ratings_df

def _get_user_vector(ratings_df: pd.DataFrame, user_id: Optional[int]) -> pd.Series:
    if user_id is None:
        return pd.Series(dtype=float)
    s = ratings_df[ratings_df['user_id'] == user_id].set_index('book_id')['rating']
    return s

def _get_seed_vector(seed_book_ids: Optional[List[int]]) -> pd.Series:
    if not seed_book_ids:
        return pd.Series(dtype=float)
    return pd.Series({int(b): 5.0 for b in seed_book_ids})

def _cosine(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return 0.0
    av = a.loc[common].astype(float).values
    bv = b.loc[common].astype(float).values
    denom = (np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0:
        return 0.0
    return float(np.dot(av, bv) / denom)

def _top_k_neighbors(ratings_df: pd.DataFrame, target_vec: pd.Series, k: int=5) -> List[Tuple[int,float]]:
    if target_vec.empty:
        return []
    tv = target_vec.sort_values(ascending=False).head(20)
    cand = ratings_df[ratings_df['book_id'].isin(tv.index)]
    overlap_counts = cand.groupby('user_id')['book_id'].nunique()
    candidate_users = overlap_counts[overlap_counts >= 2].sort_values(ascending=False).head(1500).index
    cand = cand[cand['user_id'].isin(candidate_users)]
    neighbors = []
    for uid, grp in cand.groupby('user_id'):
        vec = grp.set_index('book_id')['rating']
        sim = _cosine(tv, vec)
        if sim > 0:
            neighbors.append((int(uid), float(sim)))
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors[:k]

def _pool_from_neighbors(ratings_df: pd.DataFrame, neighbors: List[Tuple[int,float]], min_rating:int=3, pool_size:int=25) -> pd.DataFrame:
    if not neighbors:
        return pd.DataFrame(columns=['book_id','user_id','rating','sim'])
    neighbor_ids = [uid for uid,_ in neighbors]
    sim_map = {uid: sim for uid, sim in neighbors}
    df = ratings_df[ratings_df['user_id'].isin(neighbor_ids) & (ratings_df['rating'] >= min_rating)].copy()
    if df.empty:
        return df.assign(sim=0.0).head(0)
    df['sim'] = df['user_id'].map(sim_map).astype(float)
    df = df.sort_values(['rating','sim'], ascending=[False, False])
    df = df.drop_duplicates(subset=['book_id']).head(pool_size)
    return df

def _fill_with_global_top(books_df: pd.DataFrame, exclude_ids: set, need:int) -> List[Dict]:
    cols = set(books_df.columns)
    ar = 'average_rating' if 'average_rating' in cols else None
    rc = 'ratings_count' if 'ratings_count' in cols else None
    df = books_df.copy()
    if ar:
        df = df.sort_values([ar, rc] if rc else [ar], ascending=[False, False] if rc else [False])
    elif rc:
        df = df.sort_values(rc, ascending=False)
    df = df[~df['book_id'].isin(exclude_ids)]
    top = df.head(need)
    keep = [c for c in ['book_id','title','authors','image_url','average_rating','ratings_count'] if c in top.columns]
    return top[keep].to_dict(orient='records')

def _fill_with_global_avg_from_ratings(ratings_df: pd.DataFrame, books_df: pd.DataFrame, exclude_ids: set, need:int) -> List[Dict]:
    agg = ratings_df.groupby('book_id').agg(mean_rating=('rating','mean'), rating_count=('rating','count')).reset_index()
    agg = agg[~agg['book_id'].isin(exclude_ids)]
    cols_keep = [c for c in ['book_id','title','authors','image_url','average_rating','ratings_count'] if c in books_df.columns]
    info = books_df[cols_keep]
    merged = agg.merge(info, on='book_id', how='left')
    merged = merged.sort_values(['mean_rating','rating_count'], ascending=[False, False])
    top = merged.head(need)
    out = []
    for _,row in top.iterrows():
        out.append({
            'book_id': int(row['book_id']),
            'title': row.get('title'),
            'authors': row.get('authors'),
            'image_url': row.get('image_url'),
            'average_rating': row.get('average_rating', None),
            'ratings_count': row.get('ratings_count', None),
            'neighbor_count': None,
            'weighted_score': float(row['mean_rating']) if not pd.isna(row['mean_rating']) else None
        })
    return out

def _merge_book_info(books_df: pd.DataFrame, book_ids: List[int]) -> pd.DataFrame:
    cols = [c for c in ['book_id','title','authors','image_url','average_rating','ratings_count'] if c in books_df.columns]
    info = books_df[books_df['book_id'].isin(book_ids)][cols].copy()
    return info

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        user_id = payload.get("user_id", None)
        seed_book_ids = payload.get("seed_book_ids", [])
        exclude_book_ids = set(payload.get("exclude_book_ids", []))
        k = int(payload.get("k", 5))
        pool_size = int(payload.get("pool_size", 25))
        min_rating = int(payload.get("min_rating", 3))

        books_df, ratings_df = _load_books_and_ratings()

        user_vec = _get_user_vector(ratings_df, user_id) if user_id is not None else pd.Series(dtype=float)
        if user_vec.empty and seed_book_ids:
            user_vec = _get_seed_vector(seed_book_ids)

        if user_vec.empty or len(user_vec.index) == 0:
            fallback = _fill_with_global_top(books_df, exclude_book_ids, need=5)
            return jsonify({ "recommendations": fallback, "strategy": "global_top" })

        neighbors = _top_k_neighbors(ratings_df, user_vec, k=k)
        cand = _pool_from_neighbors(ratings_df, neighbors, min_rating=min_rating, pool_size=pool_size)
        cand = cand[~cand['book_id'].isin(set(user_vec.index) | exclude_book_ids)]

        # Score candidates
        picks = []
        if not cand.empty and neighbors:
            neighbor_ids = [uid for uid,_ in neighbors]
            sim_map = {uid: sim for uid, sim in neighbors}
            expanded = ratings_df[(ratings_df['user_id'].isin(neighbor_ids)) & (ratings_df['book_id'].isin(cand['book_id']))].copy()
            expanded['sim'] = expanded['user_id'].map(sim_map).astype(float)
            scored = expanded.groupby('book_id').apply(lambda g: pd.Series({
                'neighbor_count': int((g['rating']>=3).sum()),
                'weighted_score': float((g['sim'] * g['rating']).sum() / max(g['sim'].sum(), 1e-9))
            })).reset_index()
            info = _merge_book_info(books_df, scored['book_id'].tolist())
            out = scored.merge(info, on='book_id', how='left')
            out = out.sort_values(['weighted_score','neighbor_count','average_rating','ratings_count'], ascending=[False, False, False, False])
            picks = out.head(5).to_dict(orient='records')

        # Always return 5:
        chosen_ids = set(int(x['book_id']) for x in picks)
        exclude_all = exclude_book_ids | set(user_vec.index.tolist()) | chosen_ids
        deficit = 5 - len(chosen_ids)
        if deficit > 0:
            take = min(3, deficit)
            extra_avg = _fill_with_global_avg_from_ratings(ratings_df, books_df, exclude_all, need=take)
            picks += extra_avg
            chosen_ids |= set(int(x['book_id']) for x in extra_avg)
            exclude_all = exclude_book_ids | set(user_vec.index.tolist()) | chosen_ids
            deficit = 5 - len(chosen_ids)
        if deficit > 0:
            extra_top = _fill_with_global_top(books_df, exclude_all, need=deficit)
            for e in extra_top:
                e.setdefault("average_rating", None)
                e.setdefault("ratings_count", None)
                e.setdefault("neighbor_count", None)
                e.setdefault("weighted_score", None)
            picks += extra_top

        return jsonify({ "recommendations": picks, "strategy": "5NN_neighbors_then_global_fill_always_5" })

    except Exception as ex:
        return jsonify({ "error": str(ex) }), 500
# ==================== END CUSTOM RECOMMENDER ====================


def _uniq_append(dst_list, src_list, seen_ids):
    """Append records from src_list into dst_list, skipping any book_id already in seen_ids."""
    for e in src_list:
        try:
            bid = int(e.get("book_id"))
        except Exception:
            continue
        if bid not in seen_ids:
            dst_list.append(e)
            seen_ids.add(bid)


def _fallback_always_five(ratings_df, books_df, exclude_ids: set, need: int):
    """Return exactly 'need' books. Priority:
    A) Up to 3 items by global mean rating (ratings.csv), oversampled to avoid duplicates.
    B) Fill remainder by global top from books.csv metadata, oversampled.
    C) If still short, take any remaining books from books_df order.
    All while excluding IDs in exclude_ids.
    """
    picks = []
    seen = set(int(x) for x in exclude_ids)

    # A) Global mean rating from ratings.csv (community average)
    take_a = min(3, need)
    if take_a > 0 and '_fill_with_global_avg_from_ratings' in globals():
        cand_a = _fill_with_global_avg_from_ratings(ratings_df, books_df, seen, need=take_a * 3)
        _uniq_append(picks, cand_a, seen)

    # B) Global top from books.csv (metadata)
    still = need - len(picks)
    if still > 0:
        cand_b = _fill_with_global_top(books_df, seen, need=still * 3)
        _uniq_append(picks, cand_b, seen)

    # C) Last resort to guarantee
    still = need - len(picks)
    if still > 0:
        remaining = books_df[~books_df['book_id'].isin(seen)].head(still)
        keep_cols = [c for c in ['book_id','title','authors','image_url','average_rating','ratings_count'] if c in remaining.columns]
        rest = remaining[keep_cols].to_dict(orient='records')
        _uniq_append(picks, rest, seen)

    # Normalize keys
    out = []
    for e in picks[:need]:
        e.setdefault("average_rating", None)
        e.setdefault("ratings_count", None)
        e.setdefault("neighbor_count", None)
        e.setdefault("weighted_score", e.get("weighted_score", None))
        out.append(e)
    return out