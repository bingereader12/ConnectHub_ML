from flask import Flask, request, jsonify
from recommendation_model import PostRecommendationModel

app = Flask(__name__)
model = PostRecommendationModel('post_data.csv', 'user_data.csv', 'view_data.csv')

@app.route('/recommend_posts', methods=['GET'])
def recommend_posts():
    user_id = request.args.get('user_id')
    if user_id:
        recommended_posts = model.recommend_posts_for_user(user_id)
        return jsonify({'recommended_posts': recommended_posts})
    else:
        return jsonify({'error': 'Please provide a user_id in the query parameter.'})

if __name__ == '__main__':
    app.run(debug=True)
