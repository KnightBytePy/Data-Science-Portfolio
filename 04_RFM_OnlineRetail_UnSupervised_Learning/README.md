# Customer Segmentation with Unsupervised Learning (RFM + K-Means)

### TL;DR
I built an unsupervised pipeline to segment customers of an online retailer using their transaction history.  
The output is directly usable by marketing / retention teams to target high-value buyers, re-engage churn risks, and nurture new customers.

--- 

### Highlights
- Processed raw invoice-level data and engineered **RFM features**:
  - **Recency** = days since last purchase
  - **Frequency** = number of unique purchase events
  - **Monetary** = total spend
- Removed returns/credits, cleaned missing IDs, handled extreme outliers.
- Normalized features (log transform + StandardScaler) to make them clusterable.
- Used K-Means with Elbow + Silhouette to pick `k`.
- Interpreted each cluster as a real business persona (e.g. "High-Value Loyalist", "At-Risk Big Spender").
- Visualized clusters in PCA space and exported labeled customer segments.

---

📂 Data
This project uses the [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci).
Download and place it in the `data/` folder before running.
Contains 1,067,371 rows x 8 columns.

---


### Example Segments
(Your numbers will vary; these are *illustrative* based on `cluster_summary`.)

1. **High-Value Loyalists**  
   - Low Recency (they bought recently)  
   - High Frequency  
   - High Monetary  
   - Action: keep them happy with VIP perks and early access.

2. **At-Risk Big Spenders**  
   - High Monetary (historically valuable)  
   - High Recency (haven't bought in a while)  
   - Action: send win-back incentives before they churn for good.

3. **New / Low Spend**  
   - Low Frequency, low Monetary so far, recent activity.  
   - Action: nurture, educate, cross-sell.

4. **Bargain / Infrequent Buyers**  
   - Low Frequency, purchase mainly on discounts.  
   - Action: hit them with promo campaigns and sales events.

### Why this matters
- No labels required (fully unsupervised).
- Immediately actionable for CRM / lifecycle.
- Gives marketing a segmentation strategy grounded in behavior, not guesses.

### Notebook
See `Online Retail RFM - Clustering.ipynb` for the full workflow:
1. Load & clean
2. Feature engineer RFM
3. Outlier handling
4. K-Means clustering & evaluation
5. PCA visualization
6. Segment interpretation and export

