import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm

from dynamics_and_relativity.err_prop import Formula


def main() -> None:
    #loop through all 10 measurements
    for i in range(2,3):
        # get data
        data_file =f"/Users/jmeij/Downloads/OneDrive_1_9-16-2026/Results meting {i}.csv"
        output_folder = f"./results/meting{i}/"
        #distance = float(input("Totale afstand(cm):"))
        os.makedirs(output_folder, exist_ok=True)

        frame_rate = 400
        raw_data= pl.read_csv(data_file, has_header=True, separator=",", row_index_name= "frame")
        data= raw_data.select(["frame","X","Y"])
        formula = Formula("sqrt(x_change**2 + y_change**2) / frametime",vars = ["x_change","y_change"], consts={"frametime":1/frame_rate})

        #data = pl.read_csv(data_file, has_header= False,separator="\t", schema= {"x": pl.Float64, "y":pl.Float64}, row_index_name="frame")
        pixel_distance = ((data.get_column("X").item(0) - data.get_column("X").item(-1))**2 +(data.get_column("Y").item(0) - data.get_column("Y").item(-1))**2)**0.5

        #insert columns
        data.insert_column(1, ((pl.col("frame"))/frame_rate).alias("time"))
        data.insert_column(data.shape[1],(pl.col("X").diff()).alias("x_change"))
        data.insert_column(data.shape[1],(pl.col("Y").diff()).alias("y_change"))
        data.insert_column(data.shape[1],(np.sqrt(pl.col("x_change")**2 +pl.col("y_change")**2)*frame_rate).alias("velocity(cm/s)"))
        data.insert_column(data.shape[1],(pl.col("velocity(cm/s)")/100).alias("velocity(m/s)"))
        data.insert_column(data.shape[1], pl.lit(0.1).alias("delta_x_change"))
        data.insert_column(data.shape[1], pl.lit(0.1).alias("delta_y_change"))
        err_formula = formula.get_error_expr()
        print(data)
        print(err_formula)
        print(err_formula.evaluate_polars(data.drop_nans().drop_nulls()).alias("velocity"))

        # create linear fit with velocity as dependency of time.
        data = data.drop_nulls()
        time = data["time"].to_numpy()
        vel  = data["X"].to_numpy()
        Time = sm.add_constant(time)
        model = sm.OLS(vel, Time).fit()


        # save model summary to file.
        summary = model.summary(
            xname=["Intercept (x0)", "velocity (cm/s)"],
            yname="X"
        )
        with open(output_folder+"reg_sum.txt", "w") as f:
            f.write(summary.as_text())
        data = data.drop_nulls()
        time = data["time"].to_numpy()
        vel  = data["velocity(m/s)"].to_numpy()
        Time = sm.add_constant(time)
        model = sm.OLS(vel, Time).fit()


        # save model summary to file.
        summary = model.summary(
            xname=["Intercept (v0)", "velocity (m*s^-2)"],
            yname="V"
        )
        with open(output_folder+"reg_sum_v.txt", "w") as f:
            f.write(summary.as_text())


        ## create plot
        fig, ax= plt.subplots(nrows=2)
        ax[0].scatter(
            x=data["time"],
            y=data["velocity(cm/s)"],
        )
        ax[0].set(xlabel = "T (s)", ylabel="V (cm/s)")
        ax[0].grid()
        ax[0].set_title("Ball speed")
        ax[1].scatter(x=data["time"],y=data["X"])
        ax[1].set(xlabel = "T (s)", ylabel="X (cm)")
        ax[1].grid()
        ax[1].set_title("Ball place")
        plt.tight_layout()
        fig.savefig(output_folder+"plot.svg")

        ##write to output
        data.write_csv(output_folder+"/output.csv", separator = ",")

if __name__ == "__main__":
    main()
