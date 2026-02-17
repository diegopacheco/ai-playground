import { createFileRoute } from "@tanstack/react-router";
import TweetForm from "../components/TweetForm";
import TweetList from "../components/TweetList";

export const Route = createFileRoute("/")({
  component: HomePage,
});

function HomePage() {
  return (
    <div>
      <TweetForm />
      <TweetList />
    </div>
  );
}
