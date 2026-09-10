import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except:
    HAS_PLOTLY = False

st.set_page_config(page_title="Sales Forecast Pro", layout="wide", page_icon="📈")

MODEL_PATH = "best_sales_model.pkl"
ENC_PATH = "label_encoders.pkl"
FEAT_PATH = "feature_cols.pkl"
NAME_MAP_PATH = "product_store_names.csv"  # optional: Product_ID,Product_Name,Store_ID,Store_Name

CAT_COLS = ['Product_ID', 'Category', 'Store_ID', 'Store_Location', 'Day_of_Week', 'Month',
            'Quarter', 'Holiday_Name', 'Season', 'Weather', 'Sales_Channel', 'Customer_Segment',
            'Price_bin', 'Discount_bin']

DEFAULT_WEATHER = "Sunny"


# ----------------------------------------------------------------------
# feature engineering (same as the notebook)
# ----------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Holiday_Name'] = df['Holiday_Name'].fillna('No Holiday') if 'Holiday_Name' in df else 'No Holiday'

    df['Year'] = df['Date'].dt.year
    df['Month_num'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek_num'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['Month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month_num'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['DayOfWeek_num'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['DayOfWeek_num'] / 7)

    df['Price_diff_vs_competitor'] = df['Price'] - df['Competitor_Price']
    df['Price_ratio_vs_competitor'] = df['Price'] / df['Competitor_Price']
    df['Discounted_Price'] = df['Price'] * (1 - df['Discount_Percentage'] / 100)
    df['Revenue_per_unit'] = 0.0
    df['Marketing_efficiency'] = (df['Revenue'] / df['Marketing_Spend'].replace(0, np.nan)).fillna(0) \
        if 'Revenue' in df else 0.0

    price_bins = [-np.inf, 500, 2000, np.inf]
    df['Price_bin'] = pd.cut(df['Price'], bins=price_bins, labels=['Low', 'Medium', 'High'])
    df['Discount_bin'] = pd.cut(df['Discount_Percentage'], bins=[-1, 10, 25, 100],
                                 labels=['Low', 'Medium', 'High'])

    df['Product_Name_word_count'] = df['Product_Name'].astype(str).apply(lambda x: len(x.split())) \
        if 'Product_Name' in df else 0
    df['Product_Name_length'] = df['Product_Name'].astype(str).apply(len) if 'Product_Name' in df else 0

    df['Is_High_Discount'] = (df['Discount_Percentage'] > 25).astype(int)
    df['Is_Price_Above_Competitor'] = (df['Price'] > df['Competitor_Price']).astype(int)
    df['Is_Low_Stock'] = (df['Stock_Availability'] == 0).astype(int)
    df['Is_Peak_Season'] = df['Season'].isin(['Winter', 'Autumn']).astype(int)
    return df


def encode_with(df, label_encoders):
    df = df.copy()
    for c in CAT_COLS:
        le = label_encoders[c]
        df[c] = df[c].astype(str)
        known = set(le.classes_)
        df[c] = df[c].apply(lambda v: v if v in known else le.classes_[0])
        df[c + '_enc'] = le.transform(df[c])
    return df


def load_name_maps():
    """Reads an optional CSV (NAME_MAP_PATH) with columns:
       Product_ID, Product_Name, Store_ID, Store_Name
       Returns two dicts: {id: name}. If the file/columns are missing,
       the dicts are empty and callers fall back to showing the ID itself."""
    product_id_to_name, store_id_to_name = {}, {}
    if os.path.exists(NAME_MAP_PATH):
        try:
            names_df = pd.read_csv(NAME_MAP_PATH)
            if {'Product_ID', 'Product_Name'}.issubset(names_df.columns):
                pdf = names_df[['Product_ID', 'Product_Name']].dropna().drop_duplicates()
                product_id_to_name = dict(zip(pdf['Product_ID'].astype(str), pdf['Product_Name'].astype(str)))
            if {'Store_ID', 'Store_Name'}.issubset(names_df.columns):
                sdf = names_df[['Store_ID', 'Store_Name']].dropna().drop_duplicates()
                store_id_to_name = dict(zip(sdf['Store_ID'].astype(str), sdf['Store_Name'].astype(str)))
        except Exception:
            pass
    return product_id_to_name, store_id_to_name


def season_for_month(month_num):
    if month_num in (12, 1, 2):
        return "Winter"
    if month_num in (3, 4, 5):
        return "Spring"
    if month_num in (6, 7, 8):
        return "Summer"
    return "Autumn"


def run_prediction(model, feature_cols, label_encoders, date_list, common, per_day):
    # builds one row per date and predicts. also returns the encoded
    # feature table so we can run shap on it later without redoing the work
    rows = []
    for d in date_list:
        row = {
            'Date': d,
            'Price': common['price'],
            'Discount_Percentage': common['discount'],
            'Competitor_Price': common['competitor_price'],
            'Economic_Indicator': common['economic_ind'],
            'Marketing_Spend': per_day['Marketing_Spend'][d],
            'Category': common['category'],
            'Product_ID': common['product_id'],
            'Product_Name': common['product_id'],
            'Store_ID': common['store_id'],
            'Store_Location': common['store_loc'],
            'Sales_Channel': common['channel'],
            'Customer_Segment': common['segment'],
            'Month': d.strftime('%B'),
            'Quarter': f"Q{(d.month - 1) // 3 + 1}",
            'Season': season_for_month(d.month),
            'Revenue': 0, 'Units_Sold': 0,
            'Promotion_Flag': per_day['Promotion_Flag'][d],
            'Stock_Availability': per_day['Stock_Availability'][d],
            'Holiday_Flag': per_day['Holiday_Flag'][d],
            'Local_Event_Flag': per_day['Local_Event_Flag'][d],
            'Is_Weekend': per_day['Is_Weekend'][d],
            'Weather': per_day['Weather'][d],
            'Holiday_Name': "Holiday" if per_day['Holiday_Flag'][d] == 1 else np.nan,
        }
        rows.append(row)

    rows_df = pd.DataFrame(rows)
    feat = engineer_features(rows_df)
    feat = encode_with(feat, label_encoders)
    preds = model.predict(feat[feature_cols])
    preds = np.maximum(0, preds)
    result = pd.DataFrame({'Date': date_list, 'Predicted_Units_Sold': preds})
    return result, feat.reset_index(drop=True)


def mark_first_n_per_segment(segments, n):
    # marks first n days of each segment (week/month/year) as 1
    out = {}
    for seg in segments:
        for i, d in enumerate(seg):
            out[d] = 1 if i < n else 0
    return out


def spread_count_over_segments(segments, total_count):
    # same idea but total_count is one big number spread across all segments
    out = {}
    remaining = int(total_count)
    for seg in segments:
        take = max(0, min(remaining, len(seg)))
        for i, d in enumerate(seg):
            out[d] = 1 if i < take else 0
        remaining = remaining - take
    return out


def month_segments(start_year, start_month, end_year, end_month):
    segs = []
    y, m = int(start_year), int(start_month)
    while (y, m) <= (int(end_year), int(end_month)):
        start = pd.Timestamp(year=y, month=m, day=1)
        end = start + pd.offsets.MonthEnd(0)
        segs.append(pd.date_range(start, end, freq='D').tolist())
        m += 1
        if m > 12:
            m = 1
            y += 1
    return segs


def year_segments(start_year, end_year):
    segs = []
    for y in range(int(start_year), int(end_year) + 1):
        start = pd.Timestamp(year=y, month=1, day=1)
        end = pd.Timestamp(year=y, month=12, day=31)
        segs.append(pd.date_range(start, end, freq='D').tolist())
    return segs


def check_not_too_many(count, max_allowed, label):
    if count > max_allowed:
        st.error(f"{label} = {count} is more than the {max_allowed} days available. Please fix.")
        return False
    return True


def show_shap_plot(model, feature_cols, feat_row, extra_text=""):
    try:
        import shap
        import matplotlib.pyplot as plt
    except:
        st.info("Install shap and matplotlib to see why the model predicted this number.")
        return
    try:
        st.write("**Why this prediction? (SHAP)**", extra_text)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(feat_row[feature_cols])
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.write("could not make shap plot:", e)


def show_line_chart(result, title="Predicted Units Sold"):
    if HAS_PLOTLY:
        fig = px.line(result, x='Date', y='Predicted_Units_Sold', markers=True, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(result.set_index('Date')['Predicted_Units_Sold'])


def show_weekly_trend(result):
    weekly = result.set_index('Date').resample('W')['Predicted_Units_Sold'].sum().reset_index()
    weekly['Week'] = ["Week " + str(i + 1) for i in range(len(weekly))]
    if HAS_PLOTLY:
        fig = px.line(weekly, x='Week', y='Predicted_Units_Sold', markers=True, title="Weekly Sales Trend")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(weekly.set_index('Week')['Predicted_Units_Sold'])


def show_store_bar_chart(store_result, title="Predicted Units Sold by Store"):
    if HAS_PLOTLY:
        fig = px.bar(store_result, x='Store_Name', y='Predicted_Units_Sold', title=title,
                     text='Predicted_Units_Sold')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(store_result.set_index('Store_Name')['Predicted_Units_Sold'])


def show_quarterly_chart(result):
    r = result.copy()
    r['Quarter'] = r['Date'].dt.quarter
    q = r.groupby('Quarter')['Predicted_Units_Sold'].sum().reindex([1, 2, 3, 4], fill_value=0)
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Q1", "Q2", "Q3", "Q4"], y=q.values))
        fig.update_layout(title="Quarterly Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(q.rename(index={1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}))


# ----------------------------------------------------------------------
# app starts here
# ----------------------------------------------------------------------
st.title("📈 Sales Forecasting Dashboard")
st.caption("XGBoost model that predicts units sold, based on price/discount/marketing etc.")

with st.sidebar:
    st.header("Model")

model, label_encoders, feature_cols = None, None, None

if os.path.exists(MODEL_PATH) and os.path.exists(ENC_PATH) and os.path.exists(FEAT_PATH):
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENC_PATH)
    feature_cols = joblib.load(FEAT_PATH)
    st.sidebar.success("Model loaded")
else:
    st.sidebar.error("Model files not found. Put best_sales_model.pkl, label_encoders.pkl and "
                      "feature_cols.pkl in this folder.")

product_id_to_name, store_id_to_name = load_name_maps()

st.divider()
st.subheader("Make a Prediction")

if model is None:
    st.info("Model not loaded yet.")
else:
    today = pd.Timestamp.today().normalize().date()

    pred_type = st.radio("Forecast type:",
                          ["📅 Day Forecast", "📆 Week Forecast", "🗓️ Month Forecast", "📈 Year Forecast",
                           "🏬 Store/Location Forecast"],
                          horizontal=True)

    product_id_opts = sorted(label_encoders['Product_ID'].classes_.tolist()) if 'Product_ID' in label_encoders else ["P001"]
    product_display_opts = [product_id_to_name.get(pid, pid) for pid in product_id_opts]
    product_display_to_id = dict(zip(product_display_opts, product_id_opts))

    store_id_opts = sorted(label_encoders['Store_ID'].classes_.tolist()) if 'Store_ID' in label_encoders else ["S01"]
    store_display_opts = [store_id_to_name.get(sid, sid) for sid in store_id_opts]
    store_display_to_id = dict(zip(store_display_opts, store_id_opts))

    st.markdown("#### Product & Store")
    c1, c2, c3 = st.columns(3)
    with c1:
        category_opts = sorted(label_encoders['Category'].classes_.tolist()) if 'Category' in label_encoders else ["Electronics"]
        category = st.selectbox("Category", category_opts)
        product_display = st.selectbox("Product Name", product_display_opts)
        product_id = product_display_to_id[product_display]
    with c2:
        if pred_type == "🏬 Store/Location Forecast":
            st.caption("Pick the stores to compare down below ⬇️")
            store_id = None  # set per-store inside the comparison branch
        else:
            store_display = st.selectbox("Store Name", store_display_opts)
            store_id = store_display_to_id[store_display]
        store_loc_opts = sorted(label_encoders['Store_Location'].classes_.tolist()) if 'Store_Location' in label_encoders else ["Chennai"]
        store_loc = st.selectbox("Store Location", store_loc_opts)
    with c3:
        segment = st.selectbox("Customer Segment", ["Retail", "Corporate", "Wholesale"])
        channel = st.selectbox("Sales Channel", ["Online", "In-Store", "Wholesale"])

    if not product_id_to_name or not store_id_to_name:
        st.caption("ℹ️ Showing raw IDs where a name isn't available. Add a `product_store_names.csv` "
                   "(columns: Product_ID, Product_Name, Store_ID, Store_Name) next to the app to show "
                   "real names everywhere.")

    date_list = []
    selected_store_ids = []
    valid_period = True

    # ================= DAY =================
    if pred_type == "📅 Day Forecast":
        st.markdown("#### Day Details")
        d1, d2, d3 = st.columns(3)
        with d1:
            sel_date = st.date_input("Date", value=today, min_value=today)
            price = st.number_input("Price", min_value=0.0, value=1500.0)
            discount = st.slider("Discount %", 0, 100, 10)
        with d2:
            competitor_price = st.number_input("Competitor Price", min_value=0.0, value=1600.0)
            marketing_spend_total = st.number_input("Marketing Spend (Daily)", min_value=0.0, value=2000.0)
            stock_opt = st.selectbox("Stock Availability", ["No", "Yes"], index=1)
        with d3:
            promo_opt = st.selectbox("Promotion Flag", ["No", "Yes"])
            economic_ind = st.number_input("Economic Indicator", value=100.0)

        e1, e2, e3 = st.columns(3)
        with e1:
            weather = st.selectbox("Weather", ["Sunny", "Rainy"])
        with e2:
            holiday_opt = st.checkbox("This date is a Holiday")
        with e3:
            event_opt = st.checkbox("This date is a Local Event day")

        date_list = [pd.to_datetime(sel_date)]
        promo_map = {date_list[0]: 1 if promo_opt == "Yes" else 0}
        holiday_map = {date_list[0]: 1 if holiday_opt else 0}
        event_map = {date_list[0]: 1 if event_opt else 0}
        stock_map = {date_list[0]: 1 if stock_opt == "Yes" else 0}
        weather_map = {date_list[0]: weather}
        weekend_map = {date_list[0]: 1 if date_list[0].weekday() >= 5 else 0}
        spend_map = {date_list[0]: marketing_spend_total}

    # ================= WEEK =================
    elif pred_type == "📆 Week Forecast":
        st.markdown("#### Week Details")
        period = st.date_input("Forecast Period (Start - End)", value=(today, today + pd.Timedelta(days=6)), min_value=today)
        if isinstance(period, (tuple, list)) and len(period) == 2:
            week_start, week_end = period
        else:
            week_start, week_end = today, today + pd.Timedelta(days=6)
            st.info("Please select both start and end date")

        w1, w2, w3 = st.columns(3)
        with w1:
            price = st.number_input("Base Price", min_value=0.0, value=1500.0)
            discount = st.slider("Discount %", 0, 100, 10)
        with w2:
            competitor_price = st.number_input("Competitor Price", min_value=0.0, value=1600.0)
            economic_ind = st.number_input("Economic Indicator", value=100.0)
        with w3:
            marketing_spend_total = st.number_input("Weekly Marketing Budget", min_value=0.0, value=14000.0)

        p1, p2 = st.columns(2)
        with p1:
            promo_days = st.number_input("Promo Days this period (0-7 per week)", min_value=0, max_value=7, value=0, step=1)
        with p2:
            holiday_days = st.number_input("Holiday/Event Days this period (0-7 per week)", min_value=0, max_value=7, value=0, step=1)

        start_ts, end_ts = pd.to_datetime(week_start), pd.to_datetime(week_end)
        total_days = (end_ts - start_ts).days + 1

        if total_days < 1:
            st.warning("End date should be after start date")
            valid_period = False
        else:
            num_weeks = int(np.ceil(total_days / 7))
            st.caption(f"Week Count: {num_weeks}, total days: {total_days}")

            all_days = [start_ts + pd.Timedelta(days=i) for i in range(total_days)]
            week_segs = [all_days[i:i + 7] for i in range(0, total_days, 7)]
            date_list = all_days

            if not check_not_too_many(promo_days, 7, "Promo Days per week"):
                valid_period = False
            if not check_not_too_many(holiday_days, 7, "Holiday/Event Days per week"):
                valid_period = False

            promo_map = mark_first_n_per_segment(week_segs, int(promo_days))
            holiday_map = mark_first_n_per_segment(week_segs, int(holiday_days))
            event_map = {d: 0 for d in date_list}
            stock_map = {d: 1 for d in date_list}
            weather_map = {d: DEFAULT_WEATHER for d in date_list}
            weekend_map = {d: (1 if d.weekday() >= 5 else 0) for d in date_list}
            per_day_spend = marketing_spend_total / 7
            spend_map = {d: per_day_spend for d in date_list}

    # ================= MONTH =================
    elif pred_type == "🗓️ Month Forecast":
        st.markdown("#### Month Details")
        month_names = [pd.Timestamp(2020, m, 1).strftime('%B') for m in range(1, 13)]
        m1, m2, m3 = st.columns(3)
        with m1:
            sm_month = st.selectbox("Start Month", month_names, index=today.month - 1)
            sm_year = st.number_input("Start Year", min_value=today.year, max_value=today.year + 5, value=today.year)
        with m2:
            em_month = st.selectbox("End Month", month_names, index=today.month - 1)
            em_year = st.number_input("End Year", min_value=today.year, max_value=today.year + 5, value=today.year)
        with m3:
            price = st.number_input("Base Price", min_value=0.0, value=1500.0)
            discount = st.slider("Avg Discount %", 0, 100, 10)

        n1, n2 = st.columns(2)
        with n1:
            competitor_price = st.number_input("Avg Competitor Price", min_value=0.0, value=1600.0)
            promo_days = st.number_input("Promo Days per Month (0-31)", min_value=0, max_value=31, value=0, step=1)
        with n2:
            holiday_days = st.number_input("Holidays per Month (0-31)", min_value=0, max_value=31, value=0, step=1)
            marketing_spend_total = st.number_input("Monthly Marketing Budget", min_value=0.0, value=60000.0)

        economic_ind = st.number_input("Economic Indicator", value=100.0)

        sm_num = month_names.index(sm_month) + 1
        em_num = month_names.index(em_month) + 1
        start_before_today = (int(sm_year) == today.year and sm_num < today.month)
        end_before_start = (int(em_year), em_num) < (int(sm_year), sm_num)

        if start_before_today:
            st.warning("Start Month is in the past, pick current or future month")
            valid_period = False
        elif end_before_start:
            st.warning("End Month is before Start Month")
            valid_period = False
        else:
            month_segs = month_segments(sm_year, sm_num, em_year, em_num)
            date_list = [d for seg in month_segs for d in seg]
            max_month_len = max(len(seg) for seg in month_segs)

            if not check_not_too_many(promo_days, max_month_len, "Promo Days per Month"):
                valid_period = False
            if not check_not_too_many(holiday_days, max_month_len, "Holidays per Month"):
                valid_period = False

            st.caption(f"{len(month_segs)} month(s): {sm_month} {int(sm_year)} to {em_month} {int(em_year)}, "
                       f"{len(date_list)} days total.")

            promo_map = mark_first_n_per_segment(month_segs, int(promo_days))
            holiday_map = mark_first_n_per_segment(month_segs, int(holiday_days))
            event_map = {d: 0 for d in date_list}
            stock_map = {d: 1 for d in date_list}
            weather_map = {d: DEFAULT_WEATHER for d in date_list}
            weekend_map = {d: (1 if d.weekday() >= 5 else 0) for d in date_list}
            spend_map = {}
            for seg in month_segs:
                per_day_spend = marketing_spend_total / len(seg)
                for d in seg:
                    spend_map[d] = per_day_spend

    # ================= YEAR =================
    elif pred_type == "📈 Year Forecast":
        st.markdown("#### Year Details")
        y1, y2, y3 = st.columns(3)
        with y1:
            start_year = st.number_input("Start Year", min_value=today.year, max_value=today.year + 10, value=today.year)
            end_year = st.number_input("End Year", min_value=today.year, max_value=today.year + 10, value=today.year)
        with y2:
            price = st.number_input("Planned Avg Base Price", min_value=0.0, value=1500.0)
            competitor_price = st.number_input("Planned Avg Competitor Price", min_value=0.0, value=1600.0)
        with y3:
            discount = st.slider("Planned Avg Discount %", 0, 100, 10)
            economic_ind = st.number_input("Economic Indicator (e.g. GDP growth %)", value=5.0)

        marketing_spend_total = st.number_input("Yearly Marketing Budget", min_value=0.0, value=700000.0)

        if int(end_year) < int(start_year):
            st.warning("End Year is before Start Year")
            valid_period = False
        else:
            year_segs = year_segments(start_year, end_year)
            date_list = [d for seg in year_segs for d in seg]
            st.caption(f"{len(year_segs)} year(s): {int(start_year)} to {int(end_year)}, {len(date_list)} days total")

            promo_map = {d: 0 for d in date_list}
            holiday_map = {d: 0 for d in date_list}
            event_map = {d: 0 for d in date_list}
            stock_map = {d: 1 for d in date_list}
            weather_map = {d: DEFAULT_WEATHER for d in date_list}
            weekend_map = {d: (1 if d.weekday() >= 5 else 0) for d in date_list}
            spend_map = {}
            for seg in year_segs:
                per_day_spend = marketing_spend_total / len(seg)
                for d in seg:
                    spend_map[d] = per_day_spend

    # ================= STORE/LOCATION =================
    else:
        st.markdown("#### Store/Location Forecast Details")
        sel_date = st.date_input("Date", value=today, min_value=today)
        selected_stores_display = st.multiselect(
            "Compare Stores", store_display_opts,
            default=store_display_opts[:min(5, len(store_display_opts))])

        f1, f2, f3 = st.columns(3)
        with f1:
            price = st.number_input("Price", min_value=0.0, value=1500.0)
            discount = st.slider("Discount %", 0, 100, 10)
        with f2:
            competitor_price = st.number_input("Competitor Price", min_value=0.0, value=1600.0)
            marketing_spend_total = st.number_input("Marketing Spend (Daily, per store)", min_value=0.0, value=2000.0)
        with f3:
            stock_opt = st.selectbox("Stock Availability", ["No", "Yes"], index=1)
            promo_opt = st.selectbox("Promotion Flag", ["No", "Yes"])

        g1, g2, g3 = st.columns(3)
        with g1:
            weather = st.selectbox("Weather", ["Sunny", "Rainy"])
        with g2:
            holiday_opt = st.checkbox("This date is a Holiday")
        with g3:
            event_opt = st.checkbox("This date is a Local Event day")

        economic_ind = st.number_input("Economic Indicator", value=100.0)

        date_list = [pd.to_datetime(sel_date)]

        if not selected_stores_display:
            st.warning("Pick at least one store to compare")
            valid_period = False
        else:
            selected_store_ids = [store_display_to_id[s] for s in selected_stores_display]

    # ----------------- predict button -----------------
    is_store_wise = pred_type == "🏬 Store/Location Forecast"
    ready_to_predict = valid_period and (
        (is_store_wise and len(selected_store_ids) > 0) or (not is_store_wise and len(date_list) > 0)
    )

    predict_clicked = st.button("Predict", type="primary")

    if predict_clicked and ready_to_predict and is_store_wise:
        d = date_list[0]
        per_day = dict(
            Marketing_Spend={d: marketing_spend_total},
            Promotion_Flag={d: 1 if promo_opt == "Yes" else 0},
            Stock_Availability={d: 1 if stock_opt == "Yes" else 0},
            Holiday_Flag={d: 1 if holiday_opt else 0},
            Local_Event_Flag={d: 1 if event_opt else 0},
            Is_Weekend={d: 1 if d.weekday() >= 5 else 0},
            Weather={d: weather},
        )

        store_rows, feat_by_store = [], {}
        for sid in selected_store_ids:
            common = dict(price=price, discount=discount, competitor_price=competitor_price,
                          economic_ind=economic_ind, category=category, product_id=product_id,
                          store_id=sid, store_loc=store_loc, channel=channel, segment=segment)
            res, feat_enc = run_prediction(model, feature_cols, label_encoders, date_list, common, per_day)
            units = int(np.round(max(0, res['Predicted_Units_Sold'].iloc[0])))
            store_rows.append({'Store_ID': sid, 'Store_Name': store_id_to_name.get(sid, sid),
                                'Predicted_Units_Sold': units})
            feat_by_store[sid] = feat_enc

        store_result = pd.DataFrame(store_rows).sort_values('Predicted_Units_Sold', ascending=False) \
            .reset_index(drop=True)

        st.success(f"Forecast for {d.strftime('%d %b %Y')} across {len(store_result)} store(s)")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total Units (all stores)", f"{int(store_result['Predicted_Units_Sold'].sum()):,}")
        mc2.metric("Best Store", store_result.iloc[0]['Store_Name'],
                   f"{store_result.iloc[0]['Predicted_Units_Sold']} units")
        mc3.metric("Stores Compared", f"{len(store_result)}")

        show_store_bar_chart(store_result)

        with st.expander("View store-wise predictions"):
            st.dataframe(store_result[['Store_Name', 'Store_ID', 'Predicted_Units_Sold']],
                        use_container_width=True)
            st.download_button("Download store-wise predictions as CSV",
                                data=store_result.to_csv(index=False).encode("utf-8"),
                                file_name="store_wise_forecast.csv", mime="text/csv")

        best_sid = store_result.iloc[0]['Store_ID']
        show_shap_plot(model, feature_cols, feat_by_store[best_sid],
                       f"(top store: {store_result.iloc[0]['Store_Name']})")

    if predict_clicked and ready_to_predict and not is_store_wise:
        common = dict(price=price, discount=discount, competitor_price=competitor_price,
                      economic_ind=economic_ind, category=category, product_id=product_id,
                      store_id=store_id, store_loc=store_loc, channel=channel, segment=segment)
        per_day = dict(Marketing_Spend=spend_map, Promotion_Flag=promo_map, Stock_Availability=stock_map,
                       Holiday_Flag=holiday_map, Local_Event_Flag=event_map, Is_Weekend=weekend_map,
                       Weather=weather_map)

        result, feat_enc = run_prediction(model, feature_cols, label_encoders, date_list, common, per_day)
        result['Predicted_Units_Sold'] = np.round(result['Predicted_Units_Sold']).astype(int)
        n = len(date_list)

        if n == 1:
            st.success(f"Predicted Units Sold on {date_list[0].strftime('%d %b %Y')}: "
                       f"**{int(result['Predicted_Units_Sold'].iloc[0])}**")
            show_shap_plot(model, feature_cols, feat_enc.iloc[[0]])
        else:
            total = int(result['Predicted_Units_Sold'].sum())
            avg = result['Predicted_Units_Sold'].mean()
            peak_row = result.loc[result['Predicted_Units_Sold'].idxmax()]

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Predicted Units", f"{total:,}")
            mc2.metric("Average per Day", f"{avg:.1f}")
            mc3.metric("Days Covered", f"{n}")
            mc4.metric("Peak Day", peak_row['Date'].strftime('%d %b'), f"{int(peak_row['Predicted_Units_Sold'])} units")

            if pred_type == "🗓️ Month Forecast":
                show_weekly_trend(result)
            elif pred_type == "📈 Year Forecast":
                show_quarterly_chart(result)
            else:
                show_line_chart(result)

            with st.expander("View day-wise predictions"):
                st.dataframe(result, use_container_width=True)
                st.download_button("Download predictions as CSV", data=result.to_csv(index=False).encode("utf-8"),
                                    file_name="sales_forecast.csv", mime="text/csv")

            peak_idx = int(result['Predicted_Units_Sold'].idxmax())
            show_shap_plot(model, feature_cols, feat_enc.iloc[[peak_idx]], f"(peak day: {peak_row['Date'].strftime('%d %b')})")
