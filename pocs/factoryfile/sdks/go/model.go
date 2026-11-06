package factoryfile

type Factoryfile struct {
	Version   string
	Factories []Factory
}

type Factory struct {
	Name        string
	Description *string
	Entry       *string
	Agents      []Agent
	Stages      []Stage
	Loops       []Loop
}

type Agent struct {
	Name        string
	Description *string
	Role        *string
	Model       *string
	Tools       []string
	Source      *string
	Content     *string
}

type Stage struct {
	Name        string
	Description *string
	Agents      []string
	Next        []string
	Artifacts   []string
	Content     *string
}

type Loop struct {
	Name        string
	Description *string
	From        string
	To          string
	Until       *string
	Max         *int
}

func (f *Factory) Stage(name string) *Stage {
	for i := range f.Stages {
		if f.Stages[i].Name == name {
			return &f.Stages[i]
		}
	}
	return nil
}

func (f *Factory) Agent(name string) *Agent {
	for i := range f.Agents {
		if f.Agents[i].Name == name {
			return &f.Agents[i]
		}
	}
	return nil
}
