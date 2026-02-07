import styles from "./ProgressBar.module.css";

type Props = {
  progress: number;
  statusMessage: string;
};

export function ProgressBar({ progress, statusMessage }: Props) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.track}>
        <div className={styles.fill} style={{ width: `${progress}%` }} />
      </div>
      <div className={styles.percentage}>{Math.round(progress)}%</div>
      <div className={styles.status}>{statusMessage}</div>
    </div>
  );
}
