def weekly_inventory_recommendation(
    forecast_df,
    current_stock,
    safety_stock
):
    total_predicted_demand = (
        forecast_df[
            "predicted_sales"
        ].sum()
    )

    required_stock = (
        total_predicted_demand
        + safety_stock
    )

    reorder_quantity = max(
        0,
        required_stock - current_stock
    )

    if current_stock < total_predicted_demand:
        status = "Stockout Risk"

    elif current_stock < required_stock:
        status = "Low Stock"

    else:
        status = "Inventory Sufficient"

    return {
        "7_day_predicted_demand": round(
            total_predicted_demand
        ),
        "current_stock": current_stock,
        "safety_stock": safety_stock,
        "required_stock": round(
            required_stock
        ),
        "recommended_reorder": round(
            reorder_quantity
        ),
        "status": status
    }