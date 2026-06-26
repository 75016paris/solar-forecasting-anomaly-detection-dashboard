#!/usr/bin/env python3
"""Build cleaned local CSVs for the solar forecasting dashboard pipeline."""

from src.data_cleaning import build_clean_dataset


def main():
    print("=" * 80)
    print("BUILD CLEAN SOLAR DATASET")
    print("=" * 80)
    print("Note: clean/ outputs are generated from name-redacted local data.")

    outputs = build_clean_dataset('clean')

    print("\nGenerated files:")
    for name, df in outputs.items():
        print(f"  clean/{name}.csv: {len(df):,} rows")

    summary = outputs.get('data_quality_summary')
    if summary is not None and not summary.empty:
        print("\nData quality summary:")
        for _, row in summary.iterrows():
            print(
                f"  {row['plant_name']:10s} | "
                f"usable daylight: {row['usable_daylight_modeling_hours_pct']:.1f}% | "
                f"5-min completeness: {row['five_min_record_completeness_pct']:.1f}% | "
                f"warnings: {row['warnings']}"
            )

    print("\n✅ Clean dataset build complete.")


if __name__ == '__main__':
    main()
