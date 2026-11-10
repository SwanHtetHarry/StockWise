from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.forecasting import forecast_next_days
from src.inventory import weekly_inventory_recommendation


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="StockWise AI",
    layout="wide"
)


# =========================================================
# MINIMALIST CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    footer {
        visibility: hidden;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 650;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #777;
        margin-bottom: 2.5rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .status-risk {
        padding: 1rem 1.2rem;
        border: 1px solid #d6d6d6;
        border-left: 4px solid #222;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }

    .status-normal {
        padding: 1rem 1.2rem;
        border: 1px solid #d6d6d6;
        border-left: 4px solid #999;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }

    .info-card {
        border: 1px solid #e5e5e5;
        border-radius: 10px;
        padding: 1.2rem;
        height: 100%;
    }

    .info-card h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
    }

    .info-card p {
        color: #777;
        margin-bottom: 0;
        line-height: 1.5;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e6e6e6;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding-top: 0.65rem;
        padding-bottom: 0.65rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "train.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "stockwise_gradient_boosting.pkl"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_columns.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    feature_columns = joblib.load(
        FEATURES_PATH
    )

    return model, feature_columns


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


# =========================================================
# LOAD EVERYTHING
# =========================================================

with st.spinner("Loading StockWise AI..."):

    model, feature_columns = load_model()
    df = load_data()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">StockWise AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Demand forecasting and inventory planning for retail operations.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("### Forecast Settings")


store_id = st.sidebar.selectbox(
    "Store ID",
    sorted(
        df["store"].unique()
    ),
    help=(
        "The numeric ID of the store or retail location. "
        "Different stores may have different demand patterns."
    )
)


available_items = (
    df[
        df["store"] == store_id
    ]["item"]
    .unique()
)


item_id = st.sidebar.selectbox(
    "Product ID",
    sorted(
        available_items
    ),
    help=(
        "The numeric ID of the product. "
        "The dataset uses product IDs instead of product names."
    )
)


current_stock = st.sidebar.number_input(
    "Current Stock",
    min_value=0,
    value=100,
    step=1,
    help=(
        "The number of units currently available "
        "in inventory."
    )
)


safety_stock = st.sidebar.number_input(
    "Safety Stock",
    min_value=0,
    value=6,
    step=1,
    help=(
        "Extra inventory kept as protection "
        "against forecast uncertainty."
    )
)


st.sidebar.divider()

generate_forecast = st.sidebar.button(
    "Generate Forecast",
    type="primary",
    use_container_width=True
)


# =========================================================
# PRODUCT HISTORY
# =========================================================

product_history = (
    df[
        (df["store"] == store_id) &
        (df["item"] == item_id)
    ]
    .sort_values("date")
    .copy()
)


# =========================================================
# OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">Overview</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Store ID",
        store_id
    )


with col2:

    st.metric(
        "Product ID",
        item_id
    )


with col3:

    st.metric(
        "Average Daily Sales",
        f"{product_history['sales'].mean():.1f}"
    )


# =========================================================
# FORECAST SECTION
# =========================================================

if generate_forecast:

    with st.spinner(
        "Generating forecast..."
    ):

        forecast_df = forecast_next_days(
            model=model,
            df=df,
            feature_columns=feature_columns,
            store_id=store_id,
            item_id=item_id,
            days=7
        )


        recommendation = (
            weekly_inventory_recommendation(
                forecast_df=forecast_df,
                current_stock=current_stock,
                safety_stock=safety_stock
            )
        )


    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    status = recommendation[
        "status"
    ]


    if status == "Inventory Sufficient":

        st.markdown(
            f"""
            <div class="status-normal">
                Inventory Status: {status}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="status-risk">
                Inventory Status: {status}
            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # INVENTORY SUMMARY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Inventory Summary</div>',
        unsafe_allow_html=True
    )


    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        st.metric(
            "7-Day Demand",
            recommendation[
                "7_day_predicted_demand"
            ]
        )


    with metric2:

        st.metric(
            "Current Stock",
            recommendation[
                "current_stock"
            ]
        )


    with metric3:

        st.metric(
            "Required Stock",
            recommendation[
                "required_stock"
            ]
        )


    with metric4:

        st.metric(
            "Recommended Reorder",
            recommendation[
                "recommended_reorder"
            ]
        )


    # -----------------------------------------------------
    # FORECAST CHART
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">7-Day Forecast</div>',
        unsafe_allow_html=True
    )


    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        forecast_df["date"],
        forecast_df["predicted_sales"],
        marker="o",
        linewidth=2
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Predicted Sales"
    )

    ax.grid(
        alpha=0.15
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )


    # -----------------------------------------------------
    # FORECAST TABLE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Daily Forecast</div>',
        unsafe_allow_html=True
    )


    display_forecast = (
        forecast_df[
            [
                "date",
                "predicted_sales"
            ]
        ]
        .copy()
    )


    display_forecast[
        "predicted_sales"
    ] = (
        display_forecast[
            "predicted_sales"
        ]
        .round(1)
    )


    display_forecast.columns = [
        "Date",
        "Predicted Sales"
    ]


    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# HISTORICAL SALES
# =========================================================

st.markdown(
    '<div class="section-title">Recent Sales History</div>',
    unsafe_allow_html=True
)


recent_history = (
    product_history
    .tail(90)
)


fig2, ax2 = plt.subplots(
    figsize=(12, 4)
)

ax2.plot(
    recent_history["date"],
    recent_history["sales"],
    linewidth=1.5
)

ax2.set_xlabel(
    "Date"
)

ax2.set_ylabel(
    "Units Sold"
)

ax2.grid(
    alpha=0.15
)

ax2.spines[
    "top"
].set_visible(False)

ax2.spines[
    "right"
].set_visible(False)

plt.tight_layout()

st.pyplot(
    fig2
)


# =========================================================
# DATASET EXPLANATION
# =========================================================

st.markdown(
    '<div class="section-title">About the Data</div>',
    unsafe_allow_html=True
)

st.write(
    "Each row in the dataset represents daily sales for one product "
    "at one store. Store ID identifies the retail location, while "
    "Product ID identifies the product being sold. The dataset uses "
    "numeric IDs rather than real store or product names."
)


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True
)


info1, info2, info3 = st.columns(3)


with info1:

    st.markdown(
        """
        <div class="info-card">
            <h4>Forecast Demand</h4>
            <p>
            Historical sales, calendar patterns,
            lag features and rolling averages are
            analyzed by the forecasting model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        """
        <div class="info-card">
            <h4>Estimate Inventory Needs</h4>
            <p>
            The system calculates expected demand
            for the next seven days and adds the
            selected safety stock.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        """
        <div class="info-card">
            <h4>Recommend Reorder</h4>
            <p>
            Required inventory is compared with
            current stock to calculate the suggested
            reorder quantity.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "StockWise AI · Demand Forecasting and Inventory Planning"
)