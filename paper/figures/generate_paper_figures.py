"""Compatibility entrypoint for the audited figure generator."""

if __name__ == "__main__":
    from generate_all_soict_figures import generate_fig1_architecture
    from generate_figures_from_results import fig_loss, fig_roc_pr, fig_latency
    generate_fig1_architecture()
    fig_loss()
    fig_roc_pr()
    fig_latency()
    print("[+] Generated only the audited architecture and measured detector figures.")
