import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style='dark')

geolocation_df = pd.read_csv("dashboard/geolocation.csv")
items_df        = pd.read_csv("dashboard/items.csv")
orders_df       = pd.read_csv("dashboard/orders.csv")
payments_df     = pd.read_csv("dashboard/payments.csv")
products_df     = pd.read_csv("dashboard/products.csv")
reviews_df      = pd.read_csv("dashboard/reviews.csv")
customers_df    = pd.read_csv("dashboard/customers.csv")
sellers_df      = pd.read_csv("dashboard/sellers.csv")

for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
    orders_df[col] = pd.to_datetime(orders_df[col])

orders_df.sort_values("order_purchase_timestamp", inplace=True)
orders_df.reset_index(drop=True, inplace=True)

review_product = reviews_df.merge(
    items_df[['order_id', 'product_id']], on='order_id'
).merge(
    products_df[['product_id', 'product_category_name']], on='product_id'
).drop_duplicates()

def create_monthly_orders_df(orders):
    df = orders[orders['order_status'] == 'delivered'].copy()
    df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M')
    df = df.groupby('year_month').agg(order_count=('order_id', 'nunique')).reset_index()
    df['year_month'] = df['year_month'].astype(str)
    return df

def create_monthly_revenue_df(orders, payments):
    delivered = orders[orders['order_status'] == 'delivered'][['order_id', 'order_purchase_timestamp']]
    pay = payments.groupby('order_id')['payment_value'].sum().reset_index()
    merged = delivered.merge(pay, on='order_id')
    merged['year_month'] = merged['order_purchase_timestamp'].dt.to_period('M')
    df = merged.groupby('year_month').agg(revenue=('payment_value', 'sum')).reset_index()
    df['year_month'] = df['year_month'].astype(str)
    return df

def create_delivery_df(orders, customers):
    df = orders[orders['order_status'] == 'delivered'].merge(
        customers[['customer_id', 'customer_state']], on='customer_id'
    ).copy()
    df['keterlambatan'] = (
        df['order_delivered_customer_date'] - df['order_estimated_delivery_date']
    ).dt.days
    return df

def create_rfm_df(orders, payments, customers):
    delivered = orders[orders['order_status'] == 'delivered']
    pay = payments.groupby('order_id')['payment_value'].sum().reset_index()
    raw = delivered.merge(pay, on='order_id').merge(
        customers[['customer_id', 'customer_unique_id']], on='customer_id'
    )
    snapshot = raw['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = raw.groupby('customer_unique_id').agg(
        Recency=('order_purchase_timestamp', lambda x: (snapshot - x.max()).days),
        Frequency=('order_id', 'count'),
        Monetary=('payment_value', 'sum')
    ).reset_index()

    rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
    rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])

    def segment(row):
        r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])
        if r >= 3 and f >= 3 and m >= 3:
            return 'Pelanggan Setia'
        elif r >= 3 and f <= 2:
            return 'Pelanggan Baru'
        elif r <= 2 and f >= 3:
            return 'Butuh Perhatian'
        elif r == 1:
            return 'Sudah Lama Tidak Transaksi'
        else:
            return 'Pelanggan Biasa'

    rfm['Segment'] = rfm.apply(segment, axis=1)
    return rfm

def create_rfm_geo_df(rfm, customers):
    return rfm.merge(
        customers[['customer_unique_id', 'customer_city', 'customer_state']],
        on='customer_unique_id', how='left'
    )

min_date = orders_df['order_purchase_timestamp'].min().date()
max_date = orders_df['order_purchase_timestamp'].max().date()

if "start_date" not in st.session_state:
    st.session_state.start_date = min_date

if "end_date" not in st.session_state:
    st.session_state.end_date = max_date

with st.sidebar:
    st.title("Dashboard")
    st.markdown("---")

    start_date = st.date_input(
        'Tanggal Mulai',
        value=st.session_state.start_date,
        min_value=min_date,
        max_value=max_date
    )

    end_date = st.date_input(
        'Tanggal Selesai',
        value=st.session_state.end_date,
        min_value=min_date,
        max_value=max_date
    )
main_orders = orders_df[
    (orders_df['order_purchase_timestamp'].dt.date >= start_date) &
    (orders_df['order_purchase_timestamp'].dt.date <= end_date)
].copy()

monthly_orders_df  = create_monthly_orders_df(main_orders)
monthly_revenue_df = create_monthly_revenue_df(main_orders, payments_df)
delivery_df        = create_delivery_df(main_orders, customers_df)
rfm_df             = create_rfm_df(main_orders, payments_df, customers_df)
rfm_geo_df         = create_rfm_geo_df(rfm_df, customers_df)
top10_bad_reviews = review_product[review_product['review_score'] <= 2]['product_category_name'].value_counts().head(10)
avg_score         = review_product.groupby('product_category_name')['review_score'].mean()
worst_categories  = avg_score.sort_values().head(10)
payment_count     = payments_df['payment_type'].value_counts()
avg_payment       = payments_df.groupby('payment_type')['payment_value'].mean().sort_values(ascending=False)

st.header('Brazilian E-Commerce Olist Dashboard')
st.markdown("Analisis data transaksi Olist periode 2016–2018")
st.markdown("---")


st.subheader("Ringkasan Umum")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Order", value=f"{monthly_orders_df['order_count'].sum():,}")
with col2:
    st.metric("Total Revenue", value=f"R${monthly_revenue_df['revenue'].sum():,.0f}")
with col3:
    st.metric("Total Pelanggan", value=f"{main_orders['customer_id'].nunique():,}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly_orders_df['year_month'], monthly_orders_df['order_count'],
        marker='o', linewidth=2, color='steelblue')
ax.set_title('Tren Jumlah Order per Bulan (2016-2018)', fontsize=13)
ax.set_xlabel('Bulan', fontsize=11)
ax.set_ylabel('Jumlah Order (order)', fontsize=11)
ax.set_ylim(0, None)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)


# Pertanyaan 1
st.markdown("---")
st.subheader("Rata-rata Keterlambatan Pengiriman per Negara Bagian")

avg_delay = delivery_df.groupby('customer_state')['keterlambatan'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#007BFB'] + ['#5DADE2'] * (len(avg_delay) - 1)
bars = ax.bar(avg_delay.index, avg_delay.values, color=colors)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Rata-rata Keterlambatan Pengiriman per Negara Bagian')
ax.set_xlabel('Negara Bagian', fontsize=11)
ax.set_ylabel('Rata-rata Hari (+ Terlambat / - Lebih Cepat)', fontsize=11)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}', ha='center', fontsize=7)
plt.tight_layout()
ax.invert_xaxis()
st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#007BFB'] + ['#5DADE2'] * 9
    bars = ax.bar(avg_delay.head(10).index, avg_delay.head(10).values, color=colors)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Top 10 Keterlambatan Pengiriman Tertinggi')
    ax.set_xlabel('Negara Bagian', fontsize=11)
    ax.set_ylabel('Rata-rata Hari (+ Terlambat / - Lebih Cepat)', fontsize=11)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{bar.get_height():.1f}', ha='center', fontsize=7)
    plt.tight_layout()
    ax.invert_xaxis()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delivery_df['keterlambatan'].dropna(), bins=50, color='steelblue', edgecolor='white')
    ax.axvline(x=0, color='red', linewidth=1.5, linestyle='--', label='Tepat Waktu')
    ax.set_title('Distribusi Keterlambatan Pengiriman Olist')
    ax.set_xlabel('Hari (+ Terlambat / - Lebih Cepat)', fontsize=11)
    ax.set_ylabel('Jumlah Order', fontsize=11)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

with st.expander("Notes"):
    st.write("""
    - Nilai positif = terlambat, nilai negatif = lebih cepat dari estimasi.
    - Negara bagian paling biru adalah yang paling sering terlambat dan perlu perbaikan logistik.
    - Distribusi condong ke kiri berarti mayoritas pengiriman lebih cepat dari estimasi.
    """)


# Pertanyaan 2
st.markdown("---")
st.subheader("Kategori Produk dengan Review Bintang 1-2 Terbanyak")

review_product = reviews_df.merge(
    items_df[['order_id', 'product_id']], on='order_id'
).merge(
    products_df[['product_id', 'product_category_name']], on='product_id'
).drop_duplicates()

top10_bad_reviews = review_product[review_product['review_score'] <= 2]['product_category_name'].value_counts().head(10)
avg_score         = review_product.groupby('product_category_name')['review_score'].mean()
worst_categories  = avg_score.sort_values().head(10)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#007BFB'] + ['#5DADE2'] * (len(top10_bad_reviews) - 1)
    bars = ax.bar(top10_bad_reviews.index, top10_bad_reviews.values, color=colors)
    ax.set_title('Top 10 Kategori Produk dengan Review Bintang 1-2 Terbanyak')
    ax.set_xlabel('Kategori Produk', fontsize=11)
    ax.set_ylabel('Jumlah Review Buruk', fontsize=11)
    plt.xticks(rotation=40, ha='right')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(bar.get_height()), ha='center', fontsize=9)
    plt.tight_layout()
    ax.invert_xaxis()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#007BFB'] + ['#5DADE2'] * (len(worst_categories) - 1)
    ax.barh(worst_categories.index, worst_categories.values, color=colors)
    ax.axvline(x=worst_categories.mean(), color='red', linestyle='--',
               label=f'Rata-rata: {worst_categories.mean():.2f}')
    ax.set_title('Top 10 Kategori dengan Review Score Terburuk')
    ax.set_xlabel('Rata-rata Review Score', fontsize=11)
    ax.legend()
    plt.tight_layout()
    ax.invert_yaxis()
    st.pyplot(fig)

with st.expander("Notes"):
    st.write("""
    - Kategori paling biru adalah yang paling banyak mendapat keluhan dari pelanggan.
    - Kategori dengan rata-rata review score rendah perlu evaluasi kualitas produk dan pengiriman.
    """)


# Pertanyaan 3
st.markdown("---")
st.subheader("Metode Pembayaran yang Paling Sering Digunakan")

payment_count = payments_df['payment_type'].value_counts()
avg_payment   = payments_df.groupby('payment_type')['payment_value'].mean().sort_values(ascending=False)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#007BFB'] + ['#5DADE2'] * (len(payment_count) - 1)
    bars = ax.bar(payment_count.index, payment_count.values, color=colors)
    ax.set_title('Metode Pembayaran Paling Sering Digunakan Pelanggan Olist')
    ax.set_xlabel('Metode Pembayaran', fontsize=11)
    ax.set_ylabel('Jumlah Transaksi', fontsize=11)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(bar.get_height()), ha='center', fontsize=10)
    plt.tight_layout()
    ax.invert_xaxis()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#007BFB'] + ['#5DADE2'] * (len(avg_payment) - 1)
    bars = ax.bar(avg_payment.index, avg_payment.values, color=colors)
    ax.set_title('Rata-rata Nilai Transaksi per Metode Pembayaran')
    ax.set_xlabel('Metode Pembayaran', fontsize=11)
    ax.set_ylabel('Rata-rata Nilai Transaksi (BRL)', fontsize=11)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'R${bar.get_height():.0f}', ha='center', fontsize=10)
    plt.tight_layout()
    ax.invert_xaxis()
    st.pyplot(fig)

with st.expander("Notes"):
    st.write("""
    - Metode pembayaran paling biru adalah yang paling sering digunakan pelanggan Olist.
    - Metode dengan nilai transaksi tinggi bisa diprioritaskan untuk promo cashback.
    """)

# Pertanyaan 4 & 5
st.markdown("---")
st.subheader("Segmentasi Pelanggan")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rata-rata Recency (hari)", value=round(rfm_df['Recency'].mean(), 1))
with col2:
    st.metric("Rata-rata Frequency", value=round(rfm_df['Frequency'].mean(), 2))
with col3:
    st.metric("Rata-rata Monetary (BRL)", value=f"R${rfm_df['Monetary'].mean():,.0f}")

segment_count = rfm_df['Segment'].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#007BFB'] + ['#5DADE2'] * (len(segment_count) - 1)
bars = ax.bar(segment_count.index, segment_count.values, color=colors)
ax.set_title('Distribusi Segmen Pelanggan Olist')
ax.set_xlabel('Segmen', fontsize=11)
ax.set_ylabel('Jumlah Pelanggan', fontsize=11)
plt.xticks(rotation=25)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(bar.get_height()), ha='center', fontsize=10)
plt.tight_layout()
ax.invert_xaxis()
st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    monetary_segment = rfm_df.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#007BFB'] + ['#5DADE2'] * (len(monetary_segment) - 1)
    bars = ax.bar(monetary_segment.index, monetary_segment.values, color=colors)
    ax.set_title('Rata-rata Nilai Belanja per Segmen Pelanggan')
    ax.set_xlabel('Segmen', fontsize=11)
    ax.set_ylabel('Rata-rata Nilai Belanja (BRL)', fontsize=11)
    plt.xticks(rotation=25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'R${bar.get_height():.0f}', ha='center', fontsize=10)
    plt.tight_layout()
    ax.invert_xaxis()
    st.pyplot(fig)

with col2:
    segment_pct = rfm_df['Segment'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(7, 7))
    colors_pie = ['#2ecc71','#3498db','#e67e22','#e74c3c','#9b59b6']
    ax.pie(segment_pct.values, labels=segment_pct.index, autopct='%1.1f%%',
           colors=colors_pie, startangle=140)
    ax.set_title('Persentase Segmen Pelanggan Olist')
    plt.tight_layout()
    st.pyplot(fig)

churn_risk = rfm_df[rfm_df['Segment'] == 'Sudah Lama Tidak Transaksi']
st.info(f"""
**Pelanggan yang sudah lama tidak transaksi:**
- Jumlah: {len(churn_risk):,} orang
- Persentase: {round(len(churn_risk)/len(rfm_df)*100, 2)}% dari total pelanggan
- Rata-rata hari sejak transaksi terakhir: {churn_risk['Recency'].mean().round(0):.0f} hari
""")

# Pertanyaan 6
st.markdown("---")
st.subheader("Sebaran Pelanggan Paling Bernilai per Wilayah")

top_customers = rfm_geo_df[rfm_geo_df['Segment'] == 'Pelanggan Setia']

col1, col2 = st.columns(2)
with col1:
    state_count = top_customers['customer_state'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#007BFB'] + ['#5DADE2'] * (len(state_count) - 1)
    bars = ax.bar(state_count.index, state_count.values, color=colors)
    ax.set_title('Top 10 Negara Bagian dengan Pelanggan Paling Bernilai')
    ax.set_xlabel('Negara Bagian', fontsize=11)
    ax.set_ylabel('Jumlah Pelanggan Setia', fontsize=11)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(bar.get_height()), ha='center', fontsize=10)
    plt.tight_layout()
    ax.invert_xaxis()
    st.pyplot(fig)

with col2:
    monetary_state = top_customers.groupby('customer_state')['Monetary'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#007BFB'] + ['#5DADE2'] * (len(monetary_state) - 1)
    bars = ax.bar(monetary_state.index, monetary_state.values, color=colors)
    ax.set_title('Rata-rata Nilai Belanja Pelanggan Setia per Negara Bagian')
    ax.set_xlabel('Negara Bagian', fontsize=11)
    ax.set_ylabel('Rata-rata Nilai Belanja (BRL)', fontsize=11)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'R${bar.get_height():.0f}', ha='center', fontsize=10)
    plt.tight_layout()
    ax.invert_xaxis()
    st.pyplot(fig)

with st.expander("Notes"):
    st.write("""
    - Negara bagian paling biru adalah wilayah dengan konsentrasi pelanggan paling bernilai terbanyak.
    - Wilayah dengan jumlah pelanggan setia banyak sekaligus nilai belanja tinggi paling strategis bagi Olist.
    """)

st.markdown("---")
st.caption("Copyright © Pemula Dashboard")