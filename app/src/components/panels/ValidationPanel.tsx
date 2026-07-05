import { useStore } from "../../state/store";
import { Badge, EmptyState, Panel } from "../ui";

export function ValidationPanel() {
  const { validation } = useStore();
  const allPass = validation ? validation.passed === validation.total : false;
  return (
    <Panel
      index="V"
      title="Validation gates (spec §5)"
      right={validation && <Badge kind={allPass ? "badge--pass" : "badge--fail"}>{validation.passed}/{validation.total}</Badge>}
    >
      {!validation ? (
        <EmptyState text="Running gates…" />
      ) : (
        <table className="gates">
          <tbody>
            {validation.gates.map((g) => (
              <tr key={g.name}>
                <td className="gates__status">
                  <span className={"gatedot " + (g.passed ? "gatedot--ok" : "gatedot--bad")} aria-label={g.passed ? "pass" : "fail"} />
                </td>
                <td className="gates__name">{g.name}</td>
                <td className="gates__actual mono">{g.actual}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}
