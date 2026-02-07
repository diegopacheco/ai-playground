import styles from "./CodeViewer.module.css";

type Props = {
  code: string;
};

function highlightHtml(code: string): string {
  return code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(
      /(&lt;\/?)([\w-]+)/g,
      `$1<span class="${styles.tag}">$2</span>`
    )
    .replace(
      /([\w-]+)(=)/g,
      `<span class="${styles.attr}">$1</span>$2`
    )
    .replace(
      /(".*?")/g,
      `<span class="${styles.string}">$1</span>`
    );
}

export function CodeViewer({ code }: Props) {
  return (
    <div className={styles.viewer}>
      <code
        className={styles.code}
        dangerouslySetInnerHTML={{ __html: highlightHtml(code) }}
      />
    </div>
  );
}
