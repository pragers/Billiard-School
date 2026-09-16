import numpy as np
import polars as pl
import polars_ds as pds


def main() -> None:
    # get data
    data_file = input("Data:")
    distance = float(input("Totale afstand(cm):"))


    frame_rate = 120
    data = pl.read_csv(data_file, has_header= False,separator="\t", schema= {"x": pl.Float64, "y":pl.Float64}, row_index_name="frame")
    pixel_distance = ((data.get_column("x").item(0) - data.get_column("x").item(-1))**2 +(data.get_column("y").item(0) - data.get_column("y").item(-1))**2)**0.5

    #insert columns
    data.insert_column(1, (pl.col("frame")/frame_rate).alias("time"))
    data.insert_column(data.shape[1],(pl.col("x").diff()).alias("x_change"))
    data.insert_column(data.shape[1],(pl.col("y").diff()).alias("y_change"))
    data.insert_column(data.shape[1],(np.sqrt(pl.col("x_change")**2 +pl.col("y_change")**2)).alias("velocity(px/s)"))
    data.insert_column(data.shape[1],(pl.col("velocity(px/s)")/pixel_distance*(distance/100)).alias("velocity(m/s)"))


    coeffs: list = data.select(
        pds.lin_reg(pl.col("time"), target = pl.col("velocity(m/s)"),add_bias = True)
    ).item(0,0).to_list()
    result_frame = pl.from_dict({"coefficient" : coeffs[0], "constant":coeffs[1]})

    ## create plot
    plot = data.plot.scatter(x="time",y="velocity(px/s)")
    plot.save("plot.svg")

    ##write to output
    data.write_csv("./output.csv", separator = ",")
    result_frame.write_csv("./lin_reg.csv",separator=",")
